# Polygonal Marking of Plant Objects  
### Visualization of Segmentation Stages

This repository provides **visual examples of image processing stages** used in the article  
**“Automatic Polygonal Annotation of Plant Objects for Training Datasets in Green Biomass Segmentation Tasks”**.

<!-- The goal of this repository is **not to describe the full method**, but to **visually demonstrate how plant objects are segmented and converted into polygonal annotations**. -->

---

## What Is Shown in the Images

Each plant object (`crop` or `weed`) is processed independently.  
The figures illustrate the main stages of the automatic annotation pipeline.

---

### 1. Input image and object localization

- Original field image  
- Bounding boxes from Pascal VOC annotations  
- Separation of individual plant objects

**Files**
<!-- - `output_photos_EN/img_0000/0_left_original_with_boxes.png`
- `0_right_crops_on_black.png` -->

![Original image with bounding boxes](output_photos_EN/img_0000/image_0_image_1057/0_left_original_with_boxes.png)

![Cropped objects on black background](output_photos_EN/img_0000/image_0_image_1057/0_right_crops_on_black.png)
---

### 2. Initial binary mask

- Color-based segmentation in HSV space  
- Rough extraction of green biomass

<!-- **File** -->

<table>
  <tr>
    <td align="center">
      <img src="output_photos_EN/img_0000/image_0_image_1057/crop_01/3_1__visualize_mask.png" width="100%" />
      <br><b>(a)</b>
    </td>
    <td align="center">
      <img src="output_photos_EN/img_0000/image_0_image_1057/crop_01/3_2__visualize_mask.png" width="100%" />
      <br><b>(b)</b>
    </td>
    <td align="center">
      <img src="output_photos_EN/img_0000/image_0_image_1057/crop_01/3_3__visualize_mask.png" width="100%" />
      <br><b>(c)</b>
    </td>
  </tr>
</table>

---

### 3. Restored mask

- Morphological filtering  
- Noise removal and shape recovery

<!-- **File** -->
<!-- - `3_visualize_restore_mask.png` -->

<table>
  <tr>
    <td align="center">
      <img src="output_photos_EN/img_0000/image_0_image_1057/crop_01/4_1__visualize_restore_mask.png" width="100%" />
      <br><b>(a)</b>
    </td>
    <td align="center">
      <img src="output_photos_EN/img_0000/image_0_image_1057/crop_01/4_2__visualize_restore_mask.png" width="100%" />
      <br><b>(b)</b>
    </td>
    <td align="center">
      <img src="output_photos_EN/img_0000/image_0_image_1057/crop_01/4_3__visualize_restore_mask.png" width="100%" />
      <br><b>(c)</b>
    </td>
  </tr>
</table>

---

### 4. Mask evolution

- Binary masks obtained for different Hue ranges  
- Used to analyze segmentation stability

<!-- **File** -->
<!-- - `4_visualize_binary_mask_history.png` -->

![_](output_photos_EN/img_0000/image_0_image_1057/crop_01/4_visualize_binary_mask_history.png)

---

### 5. Metric dynamics

- Segmentation quality score over iterations  
- Contour area changes

These plots are used to select the optimal mask.

<!-- **Files**
- `5_visualize_score_progression.png`
- `5_visualize_area_progression.png` -->

![_](output_photos_EN/img_0000/image_0_image_1057/crop_01/5_visualize_area_progression.png)
![_](output_photos_EN/img_0000/image_0_image_1057/crop_01/5_visualize_score_progression.png)
---

### 6. Contour selection

- Contours extracted from all iterations  
- Final contour chosen before quality degradation

<!-- **File**
- `6_visualize_contour_history.png` -->

![_](output_photos_EN/img_0000/image_0_image_1057/crop_01/6_visualize_contour_history.png)

---

## Result

The selected contour is approximated by a polygon and used as an **automatic segmentation annotation** for training neural network models.


![_](https://huggingface.co/datasets/ivliev123/polygonal_marking_plant_objects/resolve/main/visualizations/image_1057_all_segmentation.jpg)

---

## Notes

- Annotations are generated automatically  
- No manual polygon labeling is required  
- Visualizations correspond directly to figures described in the article  

---

## License

Research and educational use only.  
Please cite the associated article when using this material.
