# CIS5810_Final_Project
Name: Yijie Lu, Yu Cao

Pennkey: luyijie, yucao7

Email: luyijie@seas.upenn.edu, yucao7@seas.upenn.edu

Final Project for CIS5810 at University of Pennsylvania in Fall 2023.


Before you start......

Make sure you have the following packages installed and set up. Note that you should have the same version specified in the requirements.txt.

The package segment_anything is a bit different when setting up. You should set it up following the instructions here: https://github.com/facebookresearch/segment-anything/tree/main (in the installation part)


Structure of our repo:
1. Folder 'Images': Upload your raw input images here! The input images should be named with the naming convention 'xxx_1.jpg'and 'xxx_2.jpg'. We consider the 'xxx_1.jpg' as the cameraman image, and 'xxx_2.jpg' as the group photo. (Note that, if you want to change the input images to a different style type, change the img_suffix to your desired type as well, for example '.jpeg', or '.png')

2. Folder 'result': You can find the output images here!

3. Folder 'sam_ckpt': When you set up the segment_anything model, you will have to download a file named as 'sam_vit_h_4b8939.pth'. Remember to upload this file to the folder 'sam_ckpt' before you run any notebooks. 

4. requirements.txt: Detailing the packages we used along with their versions.

5. utils.py: The functions we implemented to run by the notebooks. You do not need to modify or open this file if you want to demo a set of new photos yourself.

6. Empty_Template_for_Use.ipynb: We leave out an empty template for you to use if you want to try our model on a new set of photos of your choice. Make sure you have everything set up as specified. Follow the steps and instructions outlined in the notebook, then you will see the magic of computer vision algorithms!

7. demo.ipynb, demo2.ipynb, demo3.ipynb: These are the demos we included in our final report. You can find the input images and output images in the folder 'Images' and 'result' for your reference. 


How to run our code:
1. Set up anything as specified. (Especially the segment_anything model. Remember to upload 'sam_vit_h_4b8939.pth' to the folder 'sam_ckpt')
2. Upload the input images to the folder 'Images'.
3. Open the Empty_Template_for_Use.ipynb. Run that empty notebook following the instructions there. 
4. Find the output images in the folder 'result'.
5. Have FUN :)))))
