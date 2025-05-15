import os
from tqdm import tqdm

from semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, SemanticSAMPredictor
from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator


img_dir = '/export/data/linn/phenobench/dataset_exposed/train/images'
out_dir = "/export/data/linn/semanticSAM"

# for l in range(1,7):
if True:
    # mask_generator = SemanticSAMPredictor(build_semantic_sam(model_type='L', ckpt='/export/data/linn/semanticSAM/ckpts/swinl_only_sam_many2many.pth')) # model_type: 'L' / 'T', depends on your checkpint
    mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L', ckpt='/export/data/linn/semanticSAM/ckpts/swinl_only_sam_many2many.pth')) # model_type: 'L' / 'T', depends on your checkpint
    #mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L', ckpt='/export/data/linn/semanticSAM/ckpts/swinl_only_sam_many2many.pth'), level=[l]) # model_type: 'L' / 'T', depends on your checkpint
    for img_fn in tqdm(os.listdir(img_dir)):
        original_image, input_image = prepare_image(image_pth=os.path.join(img_dir, img_fn))  # change the image path to your image
    
        # iou_sort_masks, area_sort_masks = mask_generator.predict_masks(original_image, input_image, point=[[0.5, 0.5]]) # input point [[w, h]] relative location, i.e, [[0.5, 0.5]] is the center of the image
        # plot_multi_results(iou_sort_masks, area_sort_masks, original_image, save_path="vis", fn=img_fn)  # results and original images will be saved at save_path
    
        masks = mask_generator.generate(input_image)
        # plot_results(masks, original_image, save_path=os.path.join("alles_vis"+str(l), img_fn))  # results and original images will be saved at save_path
        # plot_results(masks, original_image, save_path=os.path.join(out_dir, "instances_"+str(l), img_fn), save_labels=True)  # results and original images will be saved at save_path
        plot_results(masks, original_image, save_path=os.path.join(out_dir, "instances_alles", img_fn), save_labels=True)  # results and original images will be saved at save_path
