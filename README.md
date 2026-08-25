## 1. Environment

### python and pytorch version

- Python 3.10.16
- Pytorch 2.1.2

### Install Dependencies

pip install -r requirements.txt

## 2. Datasets

You can refer to the following links to download the datasets.
- [LOLv1](https://daooshee.github.io/BMVC2018website/)
- LOLv2: [Baidu Pan](https://pan.baidu.com/s/17KTa-6GUUW22Q49D5DhhWw?pwd=yixu) (code: `yixu`) and  [One Drive](https://1drv.ms/u/c/2985db836826d183/EYPRJmiD24UggCmCAQAAAAABEbg62rx0FG21FwLQq0jzLg?e=Im12UA) (code: `yixu`)
- FiveK (follow [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)): [Baidu Disk](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) (code:`cyh2`), [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing)
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

## 4.Testing

```bash
# LOLv1
python eval.py --lol --perc # weights that trained with perceptual loss
python eval.py --lol # weights that trained without perceptual loss

# LOLv2-real
python eval.py --lol_v2_real --best_GT_mean # you can choose best_GT_mean or best_PSNR or best_SSIM

# LOLv2-syn
python eval.py --lol_v2_syn --perc # weights that trained with perceptual loss
python eval.py --lol_v2_syn # weights that trained without perceptual loss

# FiveK
python eval.py --fivek # output FiveK follow Retinexformer
