We present some codes to extract gameplay information (all states and actions) in Ms. Pacman from videos of gameplay. We demonstrate the code on the Atari-HEAD dataset by Zhang et al (2020) that provides the gameplay information as a series of images (AKA, "videos"). The dataset contains frame-by-frame eye data from the players (i.e., eye data on each frame as an image) and some gameplay information for each image. However, while the eye-data is very precise and of high-frequency (1000 Hz), the gameplay data is incomplete (e.g., just the actions) and contains several mistakes. 

In this work, we aimed to maximize information on task or game environment as the human players rocked the world of Ms. Pacman. Our programs identify each object (e.g., Pacman, all ghosts, their states, dots left and their positions, power pills left and their locations, bonus fruits, and more). Here is an animation of a session played by a human participant. We have put markers on the game objects () based on their estimated location. As we can see, all objects are detected with high accuracy.

<video src="anim_pacman_session_1_file_52_RZ_2394668_Aug-10-14-52-42.mp4" width="320" height="240" controls></video>

We combined the extracted gameplay data and all data from the Atari-HEAD dataset into one data file, which can be found here: https://osf.io/rd35j/. The original dataset by Zhang et al (2020) can be found here: https://zenodo.org/records/3451402.

[In progress, notebooks and codes to be uploaded soon, before July 31, 2024]

References:
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., . . . Ballard, D. (2020). Atari-HEAD: Atari human eye-tracking and demonstration dataset. In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 34, pp. 6811–6820). AAAI.
