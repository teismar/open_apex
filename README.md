# OpenApex 🏎️
**A Specialized Parser & Analytics Dashboard for RC Timing Sheets**

OpenApex is an open-source tool designed to digitize messy, scanned PDF results from Karting and RC races. It uses AI-powered OCR to convert static PDFs into interactive analytics dashboards.

## 🚀 Features
* **Messy Scan Support:** Handles photos of printouts, gray/alternating rows, and low-quality scans.
* **Dual-Session Analysis:** Upload Qualifying and Race results simultaneously.
* **Advanced Parsing:**
    * Extracts Race Results (Position, Driver, Best Time, Gap).
    * Extracts Lap Time Matrix (Lap-by-lap data for every driver).
    * Automatically filters out "Lap 0" (Out-laps) for accurate pace analysis.
* **Visual Analytics:**
    * **Pace Evolution:** Line charts showing lap time trends.
    * **Consistency:** Box plots to visualize driver consistency.
    * **Global Analysis:** Compare Qualifying pace vs. Race pace.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **OCR Engine:** RapidOCR (ONNX) - *CPU Optimized, Sub-second processing*
* **PDF Handling:** `pdf2image` + `poppler`
* **Visualization:** Plotly Express

## 📦 Installation (Docker)
The easiest way to run OpenApex is via Docker.

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/yourusername/OpenApex.git](https://github.com/yourusername/OpenApex.git)
    cd OpenApex
    ```

2.  **Run with Docker Compose**
    ```bash
    docker compose up --build
    ```

3.  **Access the App**
    Open your browser and go to `http://localhost:8501`

## 💻 Installation (Local)
If you prefer running it natively on Python 3.9+:

1.  **Install System Dependencies** (Debian/Ubuntu)
    ```bash
    sudo apt-get install poppler-utils libgl1
    ```

2.  **Install Python Packages**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 📄 License
MIT License