# Homography_Distortion_Removal - Metric Rectification

#### Description

This project aims to remove the distortions caused by Projective Homography mapping by 
* Point - Point correspondence
* Two Step Method
* One Step Method

The projective distortion is rectified upto a Similarity since the dual absolute conic is invariant to similarity.
We make use of the angles between lines and estimate the image of the dual absolute conic.

#### Dependencies

- OpenCV
- NumPy

#### Inputs

<img src="https://github.com/aartighatkesar/Homography-Distortion-Removal/blob/master/Original_Images/1.jpg" width="1555" height="1166" />  

#### Results

#### Point - Point Correspondence

_Points used for establishing one to one correspondence with world coordinates_
<br/>
<img src="https://github.com/aartighatkesar/Homography-Distortion-Removal/blob/master/Results_point_point/Results_building/p1_1.jpg" width="1555" height="1166" />  
<br/>
_After Projective Distortion removal_
<img src="https://github.com/aartighatkesar/Homography-Distortion-Removal/blob/master/Results_point_point/Results_building/p1_3.jpg" width="1555" height="1166" />  
<br/>

#### Two Step Method
_ Remove Projective distortion by mapping imaged Line at Infinity to expected [0, 0, 1].T. After this step, we are left with Affine distortion which can be estimated upto a similarity._
<br/>
<img src="https://github.com/aartighatkesar/Homography-Distortion-Removal/blob/master/Results_Two_Step_method/Results_building/two_step_buildingprojective_removalcv_2.jpg" width="447" height="382" />  
<br/>
_After Final Metric Rectification. Notice that Parallel lines are now intersecting at Line at Infinity [0, 0 ,1].T_
<img src="https://github.com/aartighatkesar/Homography-Distortion-Removal/blob/master/Results_Two_Step_method/Results_building/two_step_buildingcv_2.jpg" width="1572" height="975" />  
<br/>
 
 
#### One Step Method
_Directly estimate the image of absolute conic using five pairs of lines and angles between them._
<br/>
<img src="https://github.com/aartighatkesar/Homography-Distortion-Removal/blob/master/Results_One_step_method/Results_building/onestep_buildingcv_2.jpg" width="1555" height="1166" />  
<br/>
