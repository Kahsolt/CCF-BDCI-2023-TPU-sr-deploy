# CCF-BDCI-2023-TPU-sr-deploy

    Simplified submission code for the final reproducing verification :)

----

=> link: [https://discussion.datafountain.cn/articles/detail/3877](https://discussion.datafountain.cn/articles/detail/3877)


### method

- We adopt the [ESPCN](https://arxiv.org/abs/1609.05158) architecture as our base SR model
  - pretrained weights migrated from [https://github.com/Lornatang/ESPCN-PyTorch](https://github.com/Lornatang/ESPCN-PyTorch)
  - use `replicate` instead of `zeros` padding in Conv2d layers
  - directly process on RGB instead of YCbCr space :(
- We apply `EDGE_ENHANCE` or `SHARPEN` filter as postprocess to get better NIQE score
  - the filters can be implemented on both CPU or TPU, depending on which side is cheaper
- We develop three tiling strategies: Tile, Overlap and Crop
  - the simple `Tile` is the least time consuming (we only submit this)
  - the `Overlap` and `Crop` achieve really seamless & better visual results though
- We enable multi-threading for faster tile splitting-combining process

### run

- uncompress datasets to `dataset/`
- copy judging code to `metrics/`
- compiled bmodel
  - `cd models & bash ./convert.sh`
- install dependecies
  - `pip install -r requirements.txt`
- run evaluation
  - `cd inference & python run.py`
  - get results under `results/`

----
by Armit
2023/11/28
