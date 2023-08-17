# SAFEX - Safe City Department System

SAFEX is a comprehensive and advanced system developed using Django in Python, specifically designed to enhance the operational efficiency of the Safe City Department of Pakistan. The system empowers law enforcement agencies by providing a centralized platform to manage criminal records, criminal activities, police reports, and real-time criminal detection.

## Key Features

- **Criminal Record Management:** SAFEX allows authorized personnel to efficiently upload and manage criminal records, aiding law enforcement in tracking and identifying potential offenders.

- **Criminal Activity Repository:** The system enables seamless uploading and management of criminal activities, facilitating the collection of vital information for investigative purposes.

- **CCTV Integration:** SAFEX seamlessly integrates with CCTV cameras, enabling real-time monitoring and analysis of video streams. The integrated YOLOv7 algorithm can accurately identify and track criminals from images and videos.

- **Advanced Detection Algorithm:** The system employs the cutting-edge YOLO (You Only Look Once) algorithm, providing precise criminal detection. The transition to YOLOv8 is anticipated for even more accurate results in the future.

## Installation and Setup

1. Clone the repository to your local environment.
2. Install the required dependencies by running `pip install -r requirements.txt` and `pip install -r requirements_gpu.txt`.
3. Configure the database settings in `settings.py`.
4. Run migrations to create the necessary database tables using `python manage.py migrate`.
5. Launch the SAFEX system by executing `python manage.py runserver`.

**Note:** For security reasons, criminal images are not included in this repository.

## Usage

1. Access the SAFEX web interface through your browser.
2. Authenticate using authorized credentials.
3. Explore the intuitive dashboard to upload criminal records, manage criminal activities, and review police reports.
4. Utilize the YOLOv7 detection algorithm to analyze uploaded images and videos for criminal identification.

## Future Enhancements

- Implementation of YOLOv8 for enhanced criminal detection accuracy.
- Integration with external data sources for comprehensive criminal profiling.
- Improved user interface and data visualization tools.
- Real-time notifications and alerts for law enforcement personnel.

## Contributing

Contributions to SAFEX are welcome! If you wish to contribute, please follow the guidelines outlined in CONTRIBUTING.md.


## Contact

For inquiries and support, please contact us at [contact@safex.com](mailto:contact@safex.com).

---
Please note that this repository intentionally excludes criminal images and data for security and privacy reasons.
