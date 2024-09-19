We present a few programs to extract gameplay information (all states and actions) from videos of Atari 2600 Ms. Pacman gameplay. We test the data extraction programs on the Atari-HEAD dataset by Zhang et al. (2020), who provide the gameplay information as a series of images (or videos). Moreover, the dataset contains frame-by-frame eye data from the players (i.e., eye data on each frame as an image) and gameplay information for each image. However, while the eye-tracking data is very precise in terms time accuracy and stored at high-frequency (1000 Hz), the gameplay data acquired from the Arcade Learning Environment in Gym is incomplete (e.g., just the actions) and contains errors. 

In this work, we extracted the gameplay information directly from the gameplay videos. Our programs identify each object (e.g., Pacman, all ghosts, their states, dots left and their positions, power pills left and their locations, bonus fruits, and more). Below is an animation of a session played by a human participant. On the left, the original gameplay video is shown with the eye gaze locations overlayed as white crosshairs. On the right, the gameplay is reconstructed from the extracted information. As we see, all objects are detected with high accuracy.

https://github.com/user-attachments/assets/f184a3f5-50e4-4e98-ac19-8a7d2ff5665b

## Datasets:
We combined the extracted gameplay data and all data from the Atari-HEAD dataset into one data file, which can be found [here](https://osf.io/rd35j/). The original dataset by Zhang et al (2020) can be found in this [link]( https://zenodo.org/records/3451402).

## Brief summary of how it works:
The game objects have distinct colors, and we simply identified the objects (e.g., the Ms. Pacman avatar, the ghosts, the fruits) by their colors. Although this information is sufficient to identify the objects, several problems had to be overcome to increase the accuracy and reduce computation time.

(1) An exhaustive search for objects through all images would be very slow. Therefore, we used several computational tricks to speed up the search. For example, we first identified the maze id and restricted the search to only navigable paths on the mazes.

(2) The colors allow the objects to be detected as bunches of pixels, from which we estimate point locations by the centroids of the objects.

and more.

References:
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., . . . Ballard, D. (2020). Atari-HEAD: Atari human eye-tracking and demonstration dataset. In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 34, pp. 6811–6820). AAAI.
