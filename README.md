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
	├── FiveK
		├── test
			├──input
			├──target
		├── train
			├──input
			├──target
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
python eval.py --lol

# LOLv2-real
python eval.py --lol_v2_real

# LOLv2-syn
python eval.py --lol_v2_syn

# FiveK
python eval.py --fivek
