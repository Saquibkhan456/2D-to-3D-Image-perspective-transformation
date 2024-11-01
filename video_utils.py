import cv2
import os


def make_video_from_image_paths(folder_path, output_video='output_video.mp4', fps=30):
    # Get all jpg images in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
    print(images)
    images.sort()  # Sort the images by name (or any custom logic)

    # Check if any images were found
    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate over images and write them to video
    for image_name in images:
        print(image_name)
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error reading {image_name}")
            continue

        video.write(img)

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video}")


def make_video_from_frames(frames, output_video='output_video.mp4', fps=30):
    """
    Creates a video from a numpy array of frames.
    
    Parameters:
    frames (np.ndarray): Array of frames with shape (n, h, w) where n is the number of frames.
    output_video (str): Output video file name (default is 'output_video.mp4').
    fps (int): Frames per second (default is 30).
    """
    # Check if frames array is empty
    if frames.size == 0:
        print("No frames found in the array.")
        return

    # Get the dimensions from the first frame
    n, height, width,c = frames.shape

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=True)

    # Iterate over frames and write them to video
    for i in range(n):
        frame = frames[i]

        # Ensure the frame is in the correct shape for writing
        if frame.shape != (height, width,c):
            print(f"Skipping frame {i} due to incorrect dimensions.")
            continue

        # Convert frame to the correct format if needed
        video.write(frame)

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video}")


if __name__ == "__main__":
    make_video_from_image_paths("image_frames")

