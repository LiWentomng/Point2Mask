### Preparing datasets

```
data/  
    VOC_Panoptic/  
    coco/  
```


    
### Expected dataset structure for Pascal VOC:

```
VOC_Panoptic/  
    annotations/  
        voc_panoptic_train_aug.json    
        voc_2012_val.json  
        voc_panoptic_train_aug_1pnt_uniform/  
            2007_000032.png  
            ...  
        voc_panoptic_train_aug_10pnt_uniform/  
            2007_000032.png  
            ...  
        voc_panoptic_val/  
            2007_000033.png  
            ...  
    train/  
        2007_000032.jpg  
        ...  
    val/  
        2007_000033.jpg   
        ...
```




### Expected dataset structure for COCO2017:

```
coco/  
    annotations/   
        panoptic_train2017.json  
        panoptic_val2017.json  
        panoptic_train2017_aug_1pnt_uniform/  
        panoptic_train2017_aug_10pnt_uniform/  
        panoptic_val2017/
    train2017/  
    val2017/
```

The point label (P1 & P10) is uniform sampling from the original panoptic mask label, 
which can be download [here](https://drive.google.com/drive/folders/1XNVctOBqwXrxUSyuwec5lRIlVz-pplLX?usp=sharing). 
The training and val images for Pascal VOC can be download from [here](https://drive.google.com/file/d/16Mz13NSZBbhwPuRxiwi7ZA2Qvt9DaKtN/view?usp=sharing).









    

    
    









