# OCR

## Introduction

### files

`Burial_recognizer_form_bc.ipynb` shows pipeline of recognize burial B form step by step;	
`Deed_recognizer_form_b.ipynb` shows pipeline of recognize Deed B form step by step;	 	
`Deed_recognizer_form_c.ipynb` shows pipeline of recognize Deed C form step by step;	

### import

```python
    #coding=utf-8
    from ocr_6020 import cls_dict, text_dict
```

### API

If you want to use api to find image information, use `cls_dict()`, it will return a dictionary of image information.

```python
   cls_dict(image)
```

To recognize card, use `text_dict()`, it will return a dictionary of ocr result

```python
   text_dict(image)
```

To obtain all result, use `image_ocr()`.
```python
   image_ocr(image)
```
It will return a dictionary, like 

{'file_name': 'All_Data/MTH_LotCard_Tan_015148_01.PNG',   
'handwriting': 'N',   
'fraction': 'Y',   
'card_type': 'Deed',   
'form': 'A',   
'Name': ['DADDIO SYLIA'],   
'Lot-Sec-Gr': ['Lot No. SE 549 Section 27', 'Gr1 NE 548'],   
'Deed No. & Date': ['1/4 3 e 614 /974 ', 'Lalkd F Contract 8896 431965'],   
'Comments': ['TRANSFERRED TO]}  

## Summary
All of model in this packages are based on `EAI6020_Final_Project_Classifier.ipynb` and `EAI6020_Final_Project_Deed_Classifier.ipynb`. You **should** run both notebook first.
