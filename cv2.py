# FAKE OPENCV FOR STREAMLIT (NO LIBGL NEEDED)
# Prevents Ultralytics from crashing due to missing OpenCV dependencies

def imread(*args, **kwargs):
    return None

def imwrite(*args, **kwargs):
    return None

def imshow(*args, **kwargs):
    return None

def waitKey(*args, **kwargs):
    return None

COLOR_BGR2RGB = None
