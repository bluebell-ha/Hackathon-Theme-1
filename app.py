import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import csv
import re
import sqlite3
from datetime import datetime
import io
import os
import json
import zipfile
import base64

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="OMR Scanner")

# --- UI Styling and Background ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    /* Style for buttons */
    .stButton > button {{
        border-radius: 20px;
        border: 2px solid #FFFFFF;
        color: #FFFFFF;
        background-color: transparent;
    }}
    .stButton > button:hover {{
        border-color: #00A67D;
        color: #00A67D;
    }}
    /* Style for file uploader */
    .stFileUploader label {{
        color: white !important;
    }}
    /* Style for radio buttons to look like tabs */
    div[role="radiogroup"] > label {{
        padding: 10px 15px;
        border: 1px solid transparent;
        border-radius: 5px;
        margin: 0;
    }}
    div[role="radiogroup"] > label:hover {{
        background-color: #444;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

if os.path.exists("background.png"):
    set_background("background.png")

# --- Helper and Processing Functions (Backend Logic) ---
def sanitize_column_name(name):
    return name.lower().replace(' ', '_').replace('adv', 'advanced') + '_score'

def init_db():
    subject_cols = ", ".join([f"{sanitize_column_name(subject)} INTEGER" for subject in set(SUBJECT_MAP.values())])
    with sqlite3.connect('omr_results.db') as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, filename TEXT NOT NULL,
                omr_set TEXT NOT NULL, total_score INTEGER NOT NULL, {subject_cols}
            )
        ''')
        conn.commit()

def add_result_to_db(filename, omr_set, total_score, subject_scores):
    with sqlite3.connect('omr_results.db') as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sanitized_scores = {sanitize_column_name(k): v for k, v in subject_scores.items()}
        columns = ['timestamp', 'filename', 'omr_set', 'total_score'] + list(sanitized_scores.keys())
        placeholders = ', '.join(['?'] * len(columns))
        column_names = ', '.join(columns)
        values = [timestamp, filename, omr_set, total_score] + list(sanitized_scores.values())
        query = f"INSERT INTO results ({column_names}) VALUES ({placeholders})"
        cursor.execute(query, values)
        new_id = cursor.lastrowid
        conn.commit()
        return new_id

def get_all_results():
    try:
        with sqlite3.connect('omr_results.db') as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        st.error(f"Could not connect to the database: {e}")
        return []

def delete_record_from_db(record_id):
    with sqlite3.connect('omr_results.db') as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM results WHERE id = ?", (record_id,))
        conn.commit()

def delete_all_records():
    with sqlite3.connect('omr_results.db') as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM results")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'results'")
        conn.commit()

def display_subject_scores_as_bars(subject_scores):
    st.subheader("Subject-wise Scores:")
    for subject, score in subject_scores.items():
        if score >= 15: color = "green"
        elif score >= 8: color = "orange"
        else: color = "red"
        bar = "█" * score + "─" * (20 - score)
        st.markdown(f"**{subject}:** `{bar}` :{color}[{score}/20]")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def find_question_grid_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bubble_centers = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if 0.7 <= aspect_ratio <= 1.3 and 150 < cv2.contourArea(c) < 1000:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                bubble_centers.append((cX, cY))
    if len(bubble_centers) < 50: return None
    bubble_centers = np.array(bubble_centers)
    kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(bubble_centers[:, 0].reshape(-1, 1))
    valid_points = [p for i in range(5) if len(np.where(kmeans.labels_ == i)[0]) > 10 for p in bubble_centers[np.where(kmeans.labels_ == i)[0]]]
    if not valid_points: return None
    valid_points = np.array(valid_points)
    x_min, y_min = valid_points.min(axis=0)
    x_max, y_max = valid_points.max(axis=0)
    padding_x, padding_y = int((x_max - x_min) * 0.05), int((y_max - y_min) * 0.05)
    return np.array([[x_min - padding_x, y_min - padding_y], [x_max + padding_x, y_min - padding_y], [x_max + padding_x, y_max + padding_y], [x_min - padding_x, y_max + padding_y]], dtype="intp")

def apply_perspective_transform(image, pts):
    ordered_pts = order_points(pts)
    (tl, tr, br, bl) = ordered_pts
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    return cv2.warpPerspective(image, M, (int(width), int(height)))

def detect_omr_set_with_template(image, templates):
    if templates:
        return list(templates.keys())[0]
    return "Unknown"

@st.cache_data
def load_answer_keys_from_csv():
    keys = {}
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    files = {"A": "Key (Set A and B).xlsx - Set - A.csv", "B": "Key (Set A and B).xlsx - Set - B.csv"}
    for set_name, filename in files.items():
        try:
            with open(filename, mode='r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader) 
                key_dict = {}
                for row in reader:
                    for cell in row:
                        if not cell.strip(): continue
                        match = re.search(r'(\d+)\s*[-.]*\s*([a-dA-D])', cell)
                        if match:
                            q_num, ans = int(match.group(1)), match.group(2).upper()
                            key_dict[q_num] = answer_map.get(ans, -1)
                keys[set_name] = key_dict
        except FileNotFoundError: return None
    return keys

def identify_and_grade(warped_image, answer_key, subject_map):
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = [c for c in contours if 200 < cv2.contourArea(c) < 1200 and 0.8 < (cv2.boundingRect(c)[2] / float(cv2.boundingRect(c)[3])) < 1.2]
    scores, total_correct, flagged_count = {subject: 0 for subject in set(subject_map.values())}, 0, 0
    results_image = warped_image.copy()
    answer_bubbles = sorted([c for c in bubble_contours if cv2.boundingRect(c)[1] > warped_image.shape[0] * 0.10], key=lambda c: cv2.boundingRect(c)[1])
    
    question_rows = []
    if answer_bubbles:
        row_y = cv2.boundingRect(answer_bubbles[0])[1]
        current_row = []
        for c in answer_bubbles:
            y = cv2.boundingRect(c)[1]
            if abs(y - row_y) > cv2.boundingRect(c)[3]:
                if current_row: question_rows.append(sorted(current_row, key=lambda ctr: cv2.boundingRect(ctr)[0]))
                current_row = []
                row_y = y
            current_row.append(c)
        if current_row: question_rows.append(sorted(current_row, key=lambda ctr: cv2.boundingRect(ctr)[0]))

    for i, row in enumerate(question_rows):
        if len(row) != 20: continue
        for j in range(5):
            question_num = j * 20 + i + 1
            question_bubbles = row[j*4 : (j+1)*4]
            correct_answer_idx = answer_key.get(question_num, -1)
            
            intensities = []
            for bubble in question_bubbles:
                (x, y, w, h) = cv2.boundingRect(bubble)
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.circle(mask, (x + w//2, y + h//2), int(w * 0.4), 255, -1)
                intensities.append(cv2.mean(gray, mask=mask)[0])
            
            sorted_intensities = sorted(intensities)
            min_intensity = sorted_intensities[0]
            second_min_intensity = sorted_intensities[1]
            user_choice_idx = intensities.index(min_intensity)
            
            mean_others = np.mean([val for idx, val in enumerate(intensities) if idx != user_choice_idx]) if len(intensities) > 1 else 255
            is_marked = (min_intensity < mean_others * 0.85) and (min_intensity < 180)
            is_ambiguous = is_marked and (second_min_intensity < min_intensity * 1.15) 

            if is_marked:
                subject = subject_map.get(question_num)
                color = (0, 0, 255) # Red for incorrect
                if is_ambiguous:
                    color = (0, 255, 255) # Yellow for ambiguous
                    flagged_count += 1
                elif user_choice_idx == correct_answer_idx:
                    color = (0, 255, 0) # Green for correct
                    if subject: scores[subject] += 1
                    total_correct += 1
                
                (x_b, y_b, w_b, h_b) = cv2.boundingRect(question_bubbles[user_choice_idx])
                cv2.circle(results_image, (x_b + w_b // 2, y_b + h_b // 2), int(w_b * 0.6), color, 3)
                
    return total_correct, scores, results_image, flagged_count

def process_single_omr(image_file, templates, keys, subject_map, forced_set=None):
    try:
        image_bytes = image_file.getvalue()
        file_bytes_np = np.frombuffer(image_bytes, dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)

        if original_image is None:
            return {"status": "error", "filename": image_file.name, "message": "Could not decode image."}

        main_contour = find_question_grid_contour(original_image)
        if main_contour is None:
            return {"status": "error", "filename": image_file.name, "message": "Could not find question grid."}
        
        set_choice = forced_set
        if not set_choice:
            set_choice = detect_omr_set_with_template(original_image, templates)
        if set_choice == "Unknown" or set_choice not in keys:
             return {"status": "error", "filename": image_file.name, "message": f"Could not determine a valid OMR set."}

        answer_key = keys[set_choice]
        warped = apply_perspective_transform(original_image, main_contour)
        total, by_subject, viz_img, flagged = identify_and_grade(warped, answer_key, subject_map)
        
        new_record_id = add_result_to_db(image_file.name, set_choice, total, by_subject)
        
        session_dir = os.path.join("audit_trail", f"record_{new_record_id}")
        os.makedirs(session_dir, exist_ok=True)
        rectified_path = os.path.join(session_dir, "rectified_sheet.png")
        overlay_path = os.path.join(session_dir, "overlay.png")
        json_path = os.path.join(session_dir, "results.json")
        cv2.imwrite(rectified_path, warped)
        cv2.imwrite(overlay_path, viz_img)
        json_results = {
            "record_id": new_record_id, "filename": image_file.name, "omr_set": set_choice,
            "total_score": total, "subject_scores": by_subject, "timestamp": datetime.now().isoformat()
        }
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        return {
            "status": "success", "filename": image_file.name, "total": total, 
            "by_subject": by_subject, "viz_img": viz_img, "set_choice": set_choice,
            "new_record_id": new_record_id, "flagged": flagged,
            "audit_paths": [rectified_path, overlay_path, json_path]
        }
    except Exception as e:
        return {"status": "error", "filename": image_file.name, "message": f"An unexpected error occurred: {e}"}

# --- Main Application UI ---
SUBJECT_MAP = {i: subject for i, subject in enumerate(
    ["Python"]*20 + ["Data Analysis"]*20 + ["MySQL"]*20 +
    ["Power BI"]*20 + ["Adv Stats"]*20, 1
)}

init_db()

try:
    templates = {"A": cv2.imread("template_A.png", 0), "B": cv2.imread("template_B.png", 0)}
    keys = load_answer_keys_from_csv()
except Exception as e:
    st.error(f"Error loading initial files (keys/templates): {e}")
    templates, keys = None, None

st.markdown("<h1 style='text-align: center;'>Automated OMR Evaluation & Scoring System</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-style: italic;'>Leave the scanning to us and save your time!</div>", unsafe_allow_html=True)

tab_selection = st.radio("Navigation", ["Single Scan", "Multiple Uploads", "Dashboard & Records"], horizontal=True, label_visibility="collapsed")
st.divider()

if tab_selection == "Single Scan":
    st.header("Process a Single OMR Sheet")
    with st.expander("Usage Instructions"):
        st.info("""
            1.  **Upload or Capture:** Use the uploader or enable the camera to provide an OMR sheet image.
            2.  **Image Quality:** For best results, use a clear image with good, even lighting. Ensure the sheet is flat.
            3.  **Select Set:** Choose the correct OMR set (e.g., A or B) from the dropdown.
            4.  **Grade:** Click "Grade Sheet" to process. Results will appear on the right.
            5.  **Review:** Green circles are correct, red are incorrect, and yellow indicates an ambiguous mark (e.g., multiple bubbles filled) that may require manual review.
        """)
    st.write("---")
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("Upload or Capture OMR Sheet:")
        show_camera = st.toggle("Enable Camera Input")
        image_buffer = None
        input_filename = "camera_capture.jpg"
        if show_camera:
            cam_col, up_col = st.columns(2)
            uploaded_file = up_col.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            camera_photo = cam_col.camera_input("Take Photo", label_visibility="collapsed")
            if uploaded_file:
                image_buffer = uploaded_file
                input_filename = uploaded_file.name
            elif camera_photo:
                image_buffer = camera_photo
        else:
            uploaded_file = st.file_uploader("Upload an OMR image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file:
                image_buffer = uploaded_file
                input_filename = uploaded_file.name
        
        if 'scan_results' in st.session_state and st.session_state.scan_results:
            st.image(st.session_state.scan_results['viz_img'], caption="Grading Visualization")

    with right_col:
        if image_buffer:
            if keys and templates:
                set_options = list(keys.keys())
                set_choice = st.selectbox("Select OMR Set:", options=set_options)
                
                if st.button("Grade Sheet", type="primary"):
                    with st.spinner('Processing image...'):
                        result = process_single_omr(image_buffer, templates, keys, SUBJECT_MAP, forced_set=set_choice)
                        
                        if result["status"] == "success":
                            st.session_state.scan_results = {
                                "total": result["total"], "by_subject": result["by_subject"],
                                "viz_img": cv2.cvtColor(result["viz_img"], cv2.COLOR_BGR2RGB),
                                "new_record_id": result["new_record_id"],
                                "audit_paths": result["audit_paths"],
                                "input_filename": input_filename,
                                "set_choice": set_choice,
                                "flagged": result["flagged"]
                            }
                            st.rerun()
                        else:
                            st.error(f"Failed to process {input_filename}: {result['message']}")
            else:
                st.error("Answer keys or templates could not be loaded.")
        
        if 'scan_results' in st.session_state and st.session_state.scan_results:
            results = st.session_state.scan_results
            st.subheader("Grading Results:")
            st.metric(label="Total Score", value=f"{results['total']} / 100")
            if results['flagged'] > 0:
                st.warning(f"**{results['flagged']}** questions were flagged for ambiguous marking.")
            display_subject_scores_as_bars(results['by_subject'])
            st.success(f"Record saved to database with ID: {results['new_record_id']}")

            if st.button("Clear Screen"):
                del st.session_state.scan_results
                st.rerun()

            st.divider()
            st.subheader("Download:")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in results['audit_paths']:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            zip_buffer.seek(0)
            output = io.StringIO()
            writer = csv.writer(output)
            header = ["Filename", "OMR Set", "Total Score"] + list(results['by_subject'].keys())
            writer.writerow(header); writer.writerow([results['input_filename'], results['set_choice'], results['total']] + list(results['by_subject'].values()))
            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(label="Download Results as CSV", data=output.getvalue(), file_name=f"result_{results['input_filename'].split('.')[0]}.csv", mime='text/csv')
            dl_col2.download_button(label="Download Audit Files (.zip)", data=zip_buffer, file_name=f"audit_record_{results['new_record_id']}.zip", mime='application/zip')
        else:
            st.info("Results for the scanned sheet will be displayed here.")

elif tab_selection == "Multiple Uploads":
    st.header("Batch OMR Sheet Processing")
    uploaded_files = st.file_uploader("Upload multiple OMR sheets", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi_uploader")
    if uploaded_files:
        if not keys or not templates:
            st.error("Cannot process batch: Answer keys or templates could not be loaded.")
        else:
            set_options = list(keys.keys())
            with st.form(key="multi_set_form"):
                st.subheader("Assign OMR Set for Each File")
                assignments = {}
                for file in uploaded_files:
                    cols = st.columns([3, 2])
                    cols[0].write(file.name)
                    assignments[file] = cols[1].selectbox("Set", options=set_options, key=file.file_id)
                submitted = st.form_submit_button("Process All with Assigned Sets")

            if submitted:
                successful_grades = []; failed_grades = []
                progress_bar = st.progress(0, text="Starting batch processing...")
                for i, (file, assigned_set) in enumerate(assignments.items()):
                    progress_text = f"Processing {i+1}/{len(assignments)}: {file.name}"
                    progress_bar.progress((i + 1) / len(assignments), text=progress_text)
                    result = process_single_omr(file, templates, keys, SUBJECT_MAP, forced_set=assigned_set)
                    if result['status'] == 'success':
                        successful_grades.append({"Filename": result['filename'], "OMR Set": result['set_choice'], "Total Score": result['total'], **result['by_subject']})
                    else:
                        failed_grades.append({"Filename": result['filename'], "Reason": result['message']})

                progress_bar.empty()
                st.success(f"Batch processing complete! Processed {len(assignments)} files.")
                if successful_grades:
                    st.subheader("Mark Distribution (Successfully Graded Sheets)")
                    st.dataframe(successful_grades)
                if failed_grades:
                    st.subheader("Sheets That Failed to Process")
                    st.dataframe(failed_grades)

elif tab_selection == "Dashboard & Records":
    st.header("Evaluator Dashboard")
    results_list = get_all_results()

    if not results_list:
        st.info("The database is empty. Process an OMR sheet to see results here.")
    else:
        with st.container(border=True):
            st.subheader("Overall Performance:")
            all_scores = [item['total_score'] for item in results_list]
            
            row1_cols = st.columns(3)
            row1_cols[0].metric("Total Sheets Graded", len(results_list))
            row1_cols[1].metric("Average Total Score", f"{np.mean(all_scores):.2f}")
            row1_cols[2].metric("Best Total Score", np.max(all_scores))
            
            row2_cols = st.columns(3)
            row2_cols[0].metric("Worst Total Score", np.min(all_scores))
            row2_cols[1].metric("Median Score", f"{np.median(all_scores):.2f}")
            row2_cols[2].metric("Score Standard Deviation", f"{np.std(all_scores):.2f}")
        
        st.divider()

        st.subheader("Individual Student Performance:")
        record_options = {f"ID {res['id']}: {res['filename']} (Score: {res['total_score']})": res for res in results_list}
        selected_record_key = st.selectbox("Select a graded sheet to view details:", options=list(record_options.keys()))
        if selected_record_key:
            selected_record = record_options[selected_record_key]
            info_col, chart_col = st.columns([1, 2])
            with info_col:
                st.write(f"**Filename:** `{selected_record['filename']}`")
                st.write(f"**Graded On:** `{selected_record['timestamp']}`")
                st.metric(label=f"Total Score (Set {selected_record['omr_set']})", value=f"{selected_record['total_score']} / 100")
                st.write("")
                subject_scores_for_csv = {k.replace('_score', '').replace('_', ' ').title(): v for k, v in selected_record.items() if k.endswith('_score') and k != 'total_score'}
                output = io.StringIO()
                writer = csv.writer(output)
                header = ["Filename", "OMR Set", "Total Score"] + list(subject_scores_for_csv.keys())
                writer.writerow(header)
                writer.writerow([selected_record['filename'], selected_record['omr_set'], selected_record['total_score']] + list(subject_scores_for_csv.values()))
                csv_data = output.getvalue()
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    session_dir = os.path.join("audit_trail", f"record_{selected_record['id']}")
                    audit_files = ["rectified_sheet.png", "overlay.png", "results.json"]
                    for file_name in audit_files:
                        file_path = os.path.join(session_dir, file_name)
                        if os.path.exists(file_path):
                            zipf.write(file_path, os.path.basename(file_path))
                zip_data = zip_buffer.getvalue()
                dl_col1, dl_col2 = st.columns(2)
                dl_col1.download_button(label="Download CSV", data=csv_data, file_name=f"result_record_{selected_record['id']}.csv", mime='text/csv')
                dl_col2.download_button(label="Download Audit", data=zip_data, file_name=f"audit_record_{selected_record['id']}.zip", mime='application/zip')

            with chart_col:
                subject_scores_for_chart = {col.replace('_score', '').replace('_', ' ').title(): v for col, v in selected_record.items() if col.endswith('_score') and col != 'total_score'}
                if subject_scores_for_chart:
                    st.write("**Subject-wise Scores:**")
                    st.bar_chart(subject_scores_for_chart)
        
        st.divider()

        st.subheader("Manage Records:")
        output_all = io.StringIO()
        writer_all = csv.writer(output_all)
        if results_list:
            headers_all = list(results_list[0].keys())
            writer_all.writerow(headers_all)
            for row in results_list:
                # Ensure all values are written
                writer_all.writerow([row.get(h, '') for h in headers_all])
            st.download_button("Download All Records as CSV", data=output_all.getvalue(), file_name="all_omr_records.csv", mime="text/csv")

        display_data = [{k: v for k, v in row.items() if k != 'id'} for row in results_list]
        st.dataframe(display_data)

        delete_record_key = st.selectbox("Select a record to delete:", options=list(record_options.keys()), index=None, placeholder="Choose a record...")
        if delete_record_key and st.button("Confirm and Delete Record", type="primary"):
            record_to_delete = record_options[delete_record_key]
            delete_record_from_db(record_to_delete['id'])
            st.success(f"Record ID {record_to_delete['id']} has been deleted.")
            st.rerun()

        with st.expander("Delete All Records"):
            st.write("This will permanently delete all graded records from the database.")
            confirmation_text = st.text_input('To confirm, please type DELETE below:')
            if st.button("Delete All Records", type="primary", disabled=(confirmation_text != "DELETE")):
                delete_all_records()
                st.success("All records have been permanently deleted.")
                st.rerun()