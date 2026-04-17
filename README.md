# MFP-Net：Multi-scale Frequency and Position-aware Network for Low-Light Image Enhancement
## 1. Environment

### python and pytorch version

- Python 3.7.0
- Pytorch 1.13.1

### Install Dependencies

pip install -r requirements.txt

## 2. Datasets
Put your datasets in the following folder:
<details close> <summary>datasets (click to expand)</summary>

```
├── datasets
	├── DICM
	├── FiveK
		├── test
			├──input
			├──target
		├── train
			├──input
			├──target
	├── LIME
	├── LOLdataset
		├── our485
			├──low
			├──high
		├── eval15
			├──low
			├──high
	├── LOLv2
		├── Real_captured
			├── Train
				├── Low
				├── Normal
			├── Test
				├── Low
				├── Normal
		├── Synthetic
			├── Train
				├── Low
				├── Normal
			├── Test
				├── Low
				├── Normal
	├── LOL_blur
		├── eval
			├── high_sharp_scaled
			├── low_blur
		├── test
			├── high_sharp_scaled
				├── 0012
				├── 0017
				...
			├── low_blur
				├── 0012
				├── 0017
				...
		├── train
			├── high_sharp_scaled
				├── 0000
				├── 0001
				...
			├── low_blur
				├── 0000
				├── 0001
				...
	├── MEF
	├── NPE
	├── SICE
		├── Dataset
			├── eval
				├── target
				├── test
			├── label
			├── train
				├── 1
				├── 2
				...
		├── SICE_Grad
		├── SICE_Mix
		├── SICE_Reshape
	├── Sony_total_dark
		├── eval
			├── long
			├── short
		├── test
			├── long
				├── 10003
				├── 10006
				...
			├── short
				├── 10003
				├── 10006
				...
		├── train
			├── long
				├── 00001
				├── 00002
				...
			├── short
				├── 00001
				├── 00002
				...
	├── VV
```
</details>

## 3. Training 

- We put all the configurations that need to be adjusted in the `./data/options.py` folder and explained them in the file.

```bash
# Below is the example.
python train.py --dataset lol_v1
```

  
