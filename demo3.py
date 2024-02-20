pip3 install streamlit
pip3 install opencv-python
pip3 install pytorch



import streamlit as st
import cv2
import torch
import time

# Load YOLO model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")


# Function to get person count from a frame
def get_person_count(frame, selected_floor, selected_lift):
    results = model(frame)
    detections = []

    if not results.pandas().xyxy[0].empty:
        for result in results.pandas().xyxy[0].itertuples():
            if result[7] == "person":
                detections.append(
                    [int(result[1]), int(result[2]), int(result[3]), int(result[4])]
                )

    return len(detections)


# Main function to create the UI
def main():
    st.title("QLess")

    # Button to select floor
    selected_floor = st.sidebar.selectbox("Select Floor", list(range(-1, 11)), index=1)

    # Placeholder for displaying the count
    count_placeholder = st.empty()

    # Video capture object
    vid = cv2.VideoCapture("block1.mp4")

    # Get the selected lift
    selected_lift = st.sidebar.selectbox("Select Lift", list(range(1, 7)), index=0)

    # Button to start counting
    start_button = st.sidebar.button("Start Counting")

    # Display count in real-time if start button is clicked
    if start_button:
        while True:
            ret, frame = vid.read()

            if not ret:
                break

            # Get count of persons for the selected lift
            count = get_person_count(frame, selected_floor, selected_lift)

            # Display the count in the middle of the screen with styling
            count_placeholder.markdown(
                f"<div style='font-size: 24px; color: #3366ff; text-align: center;'>Number of Persons in Lift {selected_lift}: {count}</div>",
                unsafe_allow_html=True,
            )

            # Add a delay between frames
            time.sleep(1)


# Run the main function
if __name__ == "__main__":
    main()
