### Preparing datasets


data/  
&nbsp; &nbsp; &nbsp; VOC_Panoptic/  
&nbsp; &nbsp; &nbsp; coco/  



    
### Expected dataset structure for Pascal VOC:

VOC_Panoptic/  
&nbsp;&nbsp;&nbsp; annotations/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;voc_panoptic_train_aug.json    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;voc_2012_val.json  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;voc_panoptic_train_aug_1pnt_uniform/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2007_000032.png  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;voc_panoptic_train_aug_10pnt_uniform/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2007_000032.png  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;voc_panoptic_val/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2007_000033.png  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  


&nbsp;&nbsp;&nbsp; train/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2007_000032.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp; val/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2007_000033.jpg   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...





### Expected dataset structure for COCO2017:

coco/  
&nbsp;&nbsp;&nbsp; annotations/   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;panoptic_train2017.json  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;panoptic_val2017.json  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;panoptic_train2017_aug_1pnt_uniform/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;panoptic_train2017_aug_10pnt_uniform/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;panoptic_val2017/

&nbsp;&nbsp;&nbsp; train2017/  
&nbsp;&nbsp;&nbsp; val2017/


The point label (P1 & P10) is uniform sampling from the original panoptic mask label, 
which can be download [here](https://drive.google.com/drive/folders/1XNVctOBqwXrxUSyuwec5lRIlVz-pplLX?usp=sharing). 
The training and val images for Pascal VOC can be download from [here](https://drive.google.com/file/d/16Mz13NSZBbhwPuRxiwi7ZA2Qvt9DaKtN/view?usp=sharing).









    

    
    









