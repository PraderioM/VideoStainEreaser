This code can be used in order to select and remove stains from a video by seplacing the stain pixels with the average color of the surrounding pixels.
In order to use this code simply type the command `python -m main` in your terminal and follow the instructions on screen.
Requirements for runnin the code are python version 3.6 or greater together with the libraries specified in the `requirements.txt` file and ffmpeg library.
In order to install the python limbraries simply type `pip install -r requirements.txt` on your command line.
In order to install ffmpeg type `sudo apt install ffmpeg`.
A bad installation of the `opencv-python` library may lead to an obscure error `NULL window handler in function 'cvSetMouseCallback'`. In order to solve such error refer to the following stackoverflow post
https://stackoverflow.com/questions/62801244/null-window-handler-in-function-cvsetmousecallback

You are free to do watever the heck you feel like doing with this code.
