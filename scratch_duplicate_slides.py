import win32com.client
import os
import time

try:
    ppt_path = os.path.abspath('C:/d_drive/projects/Project1/BCLE497J Project-I_Review Presentation 1_2026.pptx')
    Application = win32com.client.Dispatch("PowerPoint.Application")
    # Open presentation in background
    Presentation = Application.Presentations.Open(ppt_path, WithWindow=False)
    
    # Duplicate Slide 5 twice to make 3 Literature Survey slides
    # Note: PowerPoint slides are 1-indexed.
    Presentation.Slides(5).Duplicate() # creates a copy at index 6
    Presentation.Slides(5).Duplicate() # creates a copy at index 6, shifting the previous one to 7
    
    Presentation.Save()
    Presentation.Close()
    Application.Quit()
    print("Successfully duplicated slides via COM.")
except Exception as e:
    print("Error:", e)
