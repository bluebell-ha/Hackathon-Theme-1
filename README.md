# Automated OMR Evaluation & Scoring System

This project is an automated, scalable OMR (Optical Mark Recognition) evaluation system built as a Streamlit web application. It is designed to digitize and accelerate the process of grading OMR sheets, reducing manual effort and providing quick, accurate results.

## Problem Statement

At Innomatics Research Labs, placement readiness assessments are regularly conducted for students. These exams use standardized OMR sheets with 100 questions, split into 5 subjects with 20 questions each. The current evaluation process is manual, where evaluators visually check each sheet. With thousands of sheets per exam, this method is:

  * **Time-Consuming:** This leads to delays in releasing results.
  * **Error-Prone:** The process is susceptible to human miscounts.
  * **Resource-Intensive:** It requires the involvement of multiple evaluators.

The objective of this project is to design and implement an automated system that can accurately evaluate OMR sheets from images captured by a mobile phone, reducing the evaluation turnaround time from days to minutes with an error tolerance of less than 0.5%.

-----

## Key Features

  * **Multi-Input Support:** Process OMR sheets from file uploads (`.png`, `.jpg`, `.jpeg`) or directly from a live camera feed.
  * **Batch Processing:** Upload and evaluate multiple OMR sheets at once, with individual set assignments.
  * **Intelligent Image Processing:** Automatically corrects for perspective distortion to ensure accurate bubble detection.
  * **Accurate Grading:** Calculates subject-wise scores (0-20 each) and a total score (0-100) by matching responses against predefined answer keys for different exam versions.
  * **Ambiguity Detection:** Flags questions with multiple marks or faint markings in yellow for manual review.
  * **Interactive Dashboard:** Provides a comprehensive dashboard with aggregate statistics and detailed views for each graded sheet.
  * **Persistent Storage:** All results are saved to a local SQLite database for tracking and review.
  * **Data Export & Auditing:**
      * Download results for a single sheet or all records as a `.csv` file.
      * For each scan, a `.zip` **audit trail** is generated, containing the rectified sheet, the graded overlay image, and a `results.json` file for complete transparency.

-----

## Tech Stack

  * **Backend & Core Logic:** Python
  * **Web Framework:** Streamlit
  * **Computer Vision:** OpenCV
  * **Numerical Operations:** NumPy
  * **Contour Analysis:** Scikit-learn (`KMeans`)
  * **Database:** SQLite
  * **Data Handling:** `csv`, `json`, `zipfile`

-----

## Technical Approach & Workflow

The application follows a robust computer vision pipeline to process and grade the OMR sheets:

1.  **Image Ingestion:** An evaluator uploads one or more OMR sheet images through the web application interface.
2.  **Contour Detection:** The system detects the sheet's orientation and main answer grid. `KMeans` clustering is used on the bubble coordinates to isolate the main answer grid from any other noise.
3.  **Perspective Correction:** Using the four corners of the detected answer grid, a perspective transform is applied to rectify any distortion. This creates a straightened, top-down view of the OMR sheet.
4.  **Bubble Grading:**
      * The warped image is processed to identify all bubbles within the grid.
      * For each question, the system classifies bubbles as marked or unmarked. It calculates the mean pixel intensity inside each bubble; the bubble with the lowest intensity (the darkest) is considered the marked answer.
      * An answer is flagged as ambiguous if the second-darkest bubble's intensity is very close to the darkest one.
5.  **Scoring & Visualization:** The identified answer is compared against the correct answer key for that OMR set version. Section-wise and total scores are calculated. A visual overlay is generated on the sheet, drawing green circles for correct answers, red for incorrect, and yellow for ambiguous ones.
6.  **Storage & Reporting:** The final scores are stored in a secure database. The audit files (rectified image, overlay image, JSON data) are saved to an `audit_trail` directory for transparency.

-----

## Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── requirements.txt        # Python dependencies
├── Key (Set A and B).xlsx - Set - A.csv  # Answer key for Set A
├── Key (Set A and B).xlsx - Set - B.csv  # Answer key for Set B
├── background.png          # Optional background image for UI
├── omr_results.db          # SQLite database (created on first run)
└── audit_trail/            # Directory for audit files (created on first run)
    └── record_1/
        ├── rectified_sheet.png
        ├── overlay.png
        └── results.json
```

-----

### Single Scan Tab

This is for grading one OMR sheet at a time.

1.  **Upload an Image:** Drag and drop an image file or click to browse.
2.  **Use Camera (Optional):** Toggle "Enable Camera Input" to take a photo directly.
3.  **Select OMR Set:** Choose the correct set (e.g., A or B) from the dropdown menu.
4.  **Grade:** Click the **"Grade Sheet"** button.
5.  **Review Results:** The total score, subject-wise breakdown, and the graded image with colored overlays will appear. You can then download the results as a CSV or a complete audit trail as a ZIP file.

### Multiple Uploads Tab

This is for batch processing.

1.  **Upload Files:** Upload all the OMR sheet images you want to process.
2.  **Assign Sets:** For each uploaded file, select its corresponding OMR set from the dropdown.
3.  **Process All:** Click the **"Process All with Assigned Sets"** button.
4.  **View Summary:** The app will display a summary table of successfully graded sheets and another table for any files that failed to process.

### Dashboard & Records Tab

This is for viewing and managing all historical data.

1.  **Overall Performance:** View aggregate metrics like total sheets graded, average score, and best/worst scores.
2.  **Inspect Individual Records:** Select a specific record from the dropdown to see its detailed scores and subject-wise performance chart.
3.  **Download Data:**
      * Download the selected record's results as a CSV or its audit files as a ZIP.
      * Use the **"Download All Records as CSV"** button to export the entire database.
4.  **Manage Records:** Select a record and click the delete button to remove it. Use the "Delete All Records" expander for a complete database wipe (with confirmation).

-----
