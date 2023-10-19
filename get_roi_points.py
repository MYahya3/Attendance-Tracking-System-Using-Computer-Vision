import cv2
# List to store clicked points
clicked_points = []
# Define the callback function to capture mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
        clicked_points.append((x, y))
        print("Clicked at (x, y) =", x, y)
# Load the image
image_path = ''
image = cv2.imread(image_path)
# Get screen dimensions
screen_width, screen_height = image.shape[1], image.shape[0]  # Adjust these values to your screen's dimensions

# Resize the image to fit the screen
image = cv2.resize(image, (screen_width, screen_height))
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# Create a window to display the image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)  # Added WINDOW_NORMAL flag

cv2.imshow('Image', image)

# Set the mouse callback function for the window
cv2.setMouseCallback('Image', mouse_callback)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print all clicked points
print("Clicked points:", clicked_points)