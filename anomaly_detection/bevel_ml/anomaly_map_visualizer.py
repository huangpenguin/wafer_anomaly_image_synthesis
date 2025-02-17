import sys

sys.path.append(".")
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from bevel_ml.data import BaseDataModule


class AnomalymapVisualizer:
    def __init__(
        self,
        datamodule: BaseDataModule,
        anomaly_maps: np.ndarray,
        save_dir: Path = None,
        separator_width: int = 5,
    ):
        self.datamodule = datamodule
        self.anomaly_maps = anomaly_maps
        self.save_dir = save_dir / "anomaly_maps"
        self.defect_classes = datamodule.image_info_df.query(
            "stage=='validate_test'"
        ).defect_class.values
        self.image_ids = datamodule.image_info_df.query(
            "stage=='validate_test'"
        ).image_id.values
        self.separator_width = separator_width

    def _adjust_gamma(self, image: np.ndarray, gamma: float = 1.0):
        table = np.array(
            [((i / 255.0) ** (1.0 / gamma) * 255) for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)

    def save_anomaly_maps(
        self, save_dir: Path | None = None, alpha: float = 0.4, gamma: float = 1.0,
        save_mask: bool = False, mask_thr: float | None = None
    ):
        if save_dir is None:
            save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        i = 0
        for batch in tqdm(
            self.datamodule.predict_dataloader(), desc="Making AnomalyMaps"
        ):
            j = 0
            images, _ = batch
            for image in images:

                idx = i * self.datamodule.batch_size + j
                image = image.numpy()
                image = (image * 255).astype(np.uint8)
                image = image.transpose(1, 2, 0)

                anomaly_map = self.anomaly_maps[idx]
                anomaly_map = (anomaly_map * 255).astype(np.uint8)
                anomaly_map = anomaly_map.transpose(1, 2, 0)# 1channel
                anomaly_map = self._adjust_gamma(anomaly_map, gamma)
            
                
                anomaly_map_colored = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
                heatmap = cv2.addWeighted(anomaly_map_colored, alpha, image, 1 - alpha, gamma)

                yellow_color = [0, 255, 255] 
                separator = np.full((image.shape[0], self.separator_width, image.shape[2]), yellow_color, dtype=image.dtype)

                result = np.concatenate((image,separator,heatmap,separator,anomaly_map_colored), axis=1)
                dst = (
                    save_dir / self.defect_classes[idx] / (self.image_ids[idx] + ".png")
                )
                if not dst.parent.exists():
                    dst.parent.mkdir(parents=True)
                cv2.imwrite(str(dst), result)
                # 異常度マップを閾値でバイナリ化した結果を保存する場合
                if save_mask:
                    anomaly_map = self.anomaly_maps[idx]
                    mask = ((anomaly_map >= mask_thr)*255).astype(np.uint8).transpose(1, 2, 0)
                    dst = (
                        save_dir.parent / "mask" / self.defect_classes[idx] / (self.image_ids[idx] + ".png")
                    )
                    if not dst.parent.exists():
                        dst.parent.mkdir(parents=True)
                    cv2.imwrite(str(dst), mask)
                j = j + 1
            i = i + 1
