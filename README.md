# ALPR - Automatic Number Plate Detection System

Python based Automatic Number Plate Recognition (ALPR) system.

## Recommended IDE Setup

[VSCode](https://code.visualstudio.com/)

## Project Setup


```sh
sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn
```

```sh
python3 -m venv venv
```

```sh
wget https://raw.githubusercontent.com/SarthakV7/AI-based-indian-license-plate-detection/master/indian_license_plate.xml
```

```sh
mkdir ./images && wget -i ./number_plate.txt -P ./images
```

### Development

```sh
python3 alpr.py
```

### Production

```sh
python3 alpr.py
```
