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

## Summary
All of model in this packages are based on `EAI6020_Final_Project_Classifier.ipynb` and `EAI6020_Final_Project_Deed_Classifier.ipynb`. You **should** run both notebook first.
