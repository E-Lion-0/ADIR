import shutil
import matplotlib
from groundingdino.util.inference import load_model, load_image, predict, annotate

import numpy as np
import torchvision.transforms as transforms
from sklearn.ensemble import IsolationForest
from torchvision.ops import box_convert
import os
from scipy.ndimage import generic_filter
import utility


# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torchvision
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor


class RailAnomalyDetector:
    def __init__(self, model_cfg_path, sam2_checkpoint, device):
        self.device = device
        self.sam2 = build_sam2(model_cfg_path, sam2_checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.sam2,
            points_per_side=32,  # Increase for more detailed segmentation
            pred_iou_thresh=0.7,  # Adjust threshold based on your needs 0.86 def
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2
        )
        self.image_predictor = SAM2ImagePredictor(self.sam2)
        self.video_predictor = build_sam2_video_predictor(model_cfg_path, sam2_checkpoint, device=device)

    def generate_masks_automatically(self, image, show=True, save=False, save_path=None, save_name=None, ret=False):
        """
        Returns:
           masks:list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """
        print(f"Generating masks...")
        masks = self.mask_generator.generate(image)
        anomalies = []
        for mask in masks:
            #if 300 < mask['point_coords'][0][1] < 450:
            #print(mask['point_coords'], mask['predicted_iou'], mask['stability_score'])
            anomalies.append(mask)

        #plt.figure(figsize=(20, 20))

        if show:
            plt.imshow(image)
            plt.axis('off')
            utility.show_anns(masks, False)
            plt.show()
        if save and save_path is not None and save_name is not None:
            plt.savefig(f"./{save_path}/{save_name[:-4]}.png")
        if ret:
            return anomalies

    def generate_mask_from_points(self, image, points, labels, show=True, save=False, save_path=None, save_name=None,
                                  return_logits=False, multimask_output=True, ret=False):
        """
        Returns:
          masks:(np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          scores:(np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          logits:(np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        self.image_predictor.set_image(image)

        masks, scores, logits = self.image_predictor.predict(
            point_coords=points,
            point_labels=labels,
            return_logits=return_logits,
            multimask_output=multimask_output,
        )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # plt.imshow(logits.squeeze(), cmap='viridis')  # Usa una colormap a tua scelta (es. 'hot', 'Reds')
        # plt.colorbar(label='Valore dei Logits')
        # plt.title("Logits della Maschera")
        # plt.axis('off')
        # plt.show()

        if show:
            utility.show_masks(image, masks, scores, point_coords=points, input_labels=labels, savefig=save,
                               save_path=save_path, save_name=save_name, show=show)
        if save and save_path is not None and save_name is not None:
            plt.savefig(f"./{save_path}/{save_name[:-4]}.png")
        if ret:
            return masks, scores, logits

    def generate_mask_from_boxes(self, image, boxes, show=True, save=False, save_path=None, save_name=None,return_logits=False,
                                 multimask_output=True, ret=False):
        """
        Returns:
          masks:(np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          scores:(np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          logits:(np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        self.image_predictor.set_image(image)
        # embedding = self.image_predictor.get_image_embedding()
        # # Flatten spatial dimensions
        # embedding_flat = embedding.reshape(256, -1).T  # Shape: (H*W, C)

        # # Apply PCA to reduce 256 channels to 3 (RGB)
        # pca = PCA(n_components=3)
        # embedding_pca = pca.fit_transform(embedding_flat)
        #
        # # Reshape back to (H, W, 3)
        # embedding_pca = embedding_pca.reshape(64, 64, 3)
        #
        # # Normalize for display
        # embedding_pca = (embedding_pca - embedding_pca.min()) / (embedding_pca.max() - embedding_pca.min())
        #
        # plt.imshow(embedding_pca)
        # plt.title("PCA Reduced Embedding")
        # plt.show()

        masks, scores, logits = self.image_predictor.predict(
            box=boxes,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        if show:
            plt.imshow(logits.squeeze(), cmap='viridis')  # Usa una colormap a tua scelta (es. 'hot', 'Reds')
            plt.colorbar(label='Valore dei Logits')
            plt.title("Logits della Maschera")
            plt.axis('off')
            plt.show()

            plt.imshow(masks[0])
            utility.show_masks(image, masks, scores, box_coords=boxes, savefig=save, save_path=save_path,
                               save_name=save_name, show=show)
        if save and save_path is not None and save_name is not None:
            plt.savefig(f"./{save_path}/{save_name[:-4]}.png")
        if ret:
            return masks, scores, logits

    def autodetect_grid(self, image, points, labels, save=False, save_path=None, save_name=None, multimask_output=True):

        ballast_mask, _, ballast_logits = self.generate_mask_from_points(image, points, labels,
                                                                         multimask_output=multimask_output, show=False,
                                                                         ret=True)
        height = int(image.shape[0] * 0.58)
        half = image[height:, :]

        object_masks = self.generate_masks_automatically(half, show=False, ret=True)
        plt.imshow(half)
        plt.axis('off')
        utility.show_anns(object_masks, False)
        plt.show()

    def autodetect_box(self, image, box, save=False, save_path=None, save_name=None, multimask_output=True):

        ballast_mask, _, ballast_logits = self.generate_mask_from_boxes(image, box, multimask_output=multimask_output,
                                                                        show=False, ret=True)
        height = int(image.shape[0] * 0.58)
        half = image[height:, :]

        object_masks = self.generate_masks_automatically(half, show=False, ret=True)
        plt.imshow(half)
        plt.axis('off')
        utility.show_anns(object_masks, False)
        plt.show()

    def cv_analysis(self, file_name, type):
        anomaly_mask = None
        # Carica l'immagine termica pre-processata (scala 0-255)
        roi_thermal = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        height = int(roi_thermal.shape[0] * 0.58)
        roi_thermal = roi_thermal[height:, :]

        if type == 'temperatura':
            #Thresholding su temperatura

            # Definiamo una soglia per il calore anomalo (es. 200 su scala 0-255)
            T_THRESHOLD = 200

            # Binarizzazione per evidenziare anomalie più calde
            _, anomaly_mask = cv2.threshold(roi_thermal, T_THRESHOLD, 255, cv2.THRESH_BINARY)


        elif type == 'statistica locale':
            # Funzione per calcolare la media locale
            def local_mean_std(image, size=15):
                mean = generic_filter(image, np.mean, size=size)
                std = generic_filter(image, np.std, size=size)
                return mean, std

            mean_local, std_local = local_mean_std(roi_thermal)

            # Pixel fuori dalla deviazione standard locale
            anomaly_mask = np.where(abs(roi_thermal - mean_local) > 2 * std_local, 255, 0).astype(np.uint8)
        elif type == 'contorni':
            # Applichiamo Canny per rilevare i bordi delle anomalie
            edges = cv2.Canny(roi_thermal, 50, 150)

            # Troviamo i contorni
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Disegniamo i contorni sulle anomalie
            anomaly_mask = roi_thermal.copy()
            # for contour in contours:
            #     area = cv2.contourArea(contour)
            #     if 300 < area < 5000:  # Filtra anomalie molto piccole o troppo grandi
            #         cv2.drawContours(anomaly_mask, [contour], -1, (255, 0, 0), 2)
            cv2.drawContours(anomaly_mask, contours, -1, (255, 0, 0), 2)

        # Visualizzazione
        if anomaly_mask is not None:
            cv2.imshow("Analisi " + type, anomaly_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def video_analysis(self, video_dir, segmented_video_dir, dino_model):
        try:

            #Extract frames
            frame_names = [
                p for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            #Sorting frames
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))
            #Setting the inference_state
            inference_state = self.video_predictor.init_state(video_path=video_dir)

            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

            # ## Grounding DINO - Finding objects in the bounding box
            dino_boxes, phrases, dino_score = grounding_Dino_analyzer(video_dir+'/'+frame_names[0],dino_model,
                                    'railway . object .', device)
            max_score_railway = 0
            railway_box = None
            object_points = []
            for i, phrase in enumerate(phrases):
                if phrase == 'railway' and dino_score[i] > max_score_railway:
                    railway_box = dino_boxes[i]
                    max_score_railway = dino_score[i]
                if phrase == 'object':
                    x_min, y_min, x_max, y_max = dino_boxes[i]
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2
                    object_points.append([x_center, y_center])

            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=railway_box,
            )
            id_objects = []
            for obj_point in object_points:
                print('obj_point ', obj_point)
                ann_obj_id += 1
                id_objects.append(ann_obj_id)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=[obj_point],
                    labels= np.array([1], np.int32),
                )

            idx_frame = 0
            video_segments = {}  # video_segments contains the per-frame segmentation results

            while idx_frame < len(frame_names):
                last_masks = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state, start_frame_idx=idx_frame):

                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }


                    # plt.close('all')
                    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
                    #
                    # for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    #     utility.show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)
                    #
                    # plt.show()

                    idx_frame += 1
                    if idx_frame % 15 == 0:
                        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                            last_masks[out_obj_id] = out_mask
                        break

                #Riutilizziamo Grounding DINO ogni 15 frame per capire se ci sono nuovi oggetti
                obj_boxes, _, obj_scores= grounding_Dino_analyzer(video_dir+'/'+frame_names[idx_frame-1], dino_model,'object .',device)
                new_boxes = []
                #controlliamo se i box contengono maschere già segmentate
                for obj_box in obj_boxes:
                    contained = False
                    for idx,mask in last_masks.items():
                        print('lll', mask.shape)
                        if is_mask_in_box(mask, obj_box):
                            print("C'era di già")
                            contained = True
                            break
                    if contained:
                        continue
                    else:
                        new_boxes.append(obj_box)



                for idx_frame_proc in range(idx_frame - 15, idx_frame):
                    plt.close('all')
                    plt.imshow(Image.open(os.path.join(video_dir, frame_names[idx_frame_proc])))

                    for out_obj_id, out_mask in video_segments[idx_frame_proc].items():
                        utility.show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)
                    plt.savefig(os.path.join(segmented_video_dir, frame_names[idx_frame_proc]),
                                bbox_inches='tight', pad_inches=0)
                    #plt.show()

                for obj_box in new_boxes:
                    print("new box ", obj_box)
                    ann_obj_id += 1

                    x_min, y_min, x_max, y_max = obj_box
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2

                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx= idx_frame,
                        obj_id=ann_obj_id,
                        points=[[x_center, y_center]],
                        labels=np.array([1], np.int32),
                    )
                # segmented_img,_ = draw_mask_2(np.array(Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))), out_mask_logits[0].cpu().numpy())
                # im = Image.fromarray(segmented_img)
                # save_path = os.path.join(segmented_video_dir, f"frame_{out_frame_idx:03d}.png")
                # im.save(save_path)# max_logit_value = out_mask_logits[0].max().item()


                # plt.imshow(out_mask_logits.squeeze(), cmap='viridis')
                # plt.colorbar(label='Valore dei Logits')
                # plt.title("Logits della Maschera")
                # plt.axis('off')
                # plt.show()
                # if out_frame_idx == 2:
                #     print("Annotiamo l'anomalia")
                #     # Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
                #     pp = np.array([[550, 370]], dtype=np.float32)
                #     # for labels, `1` means positive click and `0` means negative click
                #     ll = np.array([0], np.int32)
                #     _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                #         inference_state=inference_state,
                #         frame_idx=out_frame_idx,
                #         obj_id=ann_obj_id,
                #         points=pp,
                #         labels=ll,
                #     )
                # plt.figure(figsize=(9, 6))
                # plt.title(f"frame {out_frame_idx}")
                # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
                # utility.show_mask_v((out_mask_logits[0] > threshold).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
                # if out_frame_idx == 100:
                #     plt.close('all')



            # vis_frame_stride = 1
            # plt.close("all")
            # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            #     # image = np.asarray(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            #         utility.show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)
            #         # segmented_img = draw_mask(image, out_mask)
            #         # im = Image.fromarray(segmented_img)
            #         # save_path = os.path.join(segmented_video_dir, f"frame_{out_frame_idx:03d}.png")
            #         # im.save(save_path)
            #     plt.savefig(os.path.join(segmented_video_dir, frame_names[out_frame_idx]))
        except(KeyboardInterrupt, SystemExit):
            print("Exiting...")



    def detect_anomaly_in_image(self, image, mask, scores, logits, save_dir, file_name):

        # segmented_img, _ = draw_mask_2(image, masks)
        # im = Image.fromarray(segmented_img)
        # save_path = os.path.join(save_dir, file_name[:-4]+".jpg")
        # im.save(save_path)
        # ------------------------------
        # STEP 2: Estrazione della ROI tramite la maschera
        # ------------------------------
        mask = mask.astype(np.uint8)
        # Applica la maschera per ottenere la ROI: questo isola le regioni dei binari (e la massicciata)
        roi = cv2.bitwise_and(image, image, mask=mask)

        # Trova il bounding box attorno alla ROI per ridurre la porzione di immagine da analizzare
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        roi_cropped = roi[y:y + h, x:x + w]
        logits_cropped = logits[y:y + h, x:x + w]

        # Per DINOv2, convertiamo la ROI in un'immagine a 3 canali
        roi_cropped_color = cv2.cvtColor(roi_cropped, cv2.COLOR_GRAY2BGR)

        # ------------------------------
        # STEP 3: Configurazione per l'estrazione delle feature patch-wise con DINOv2
        # ------------------------------

        # Definisci la trasformazione per preparare le patch per DINOv2
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Dimensione di input richiesta dal modello
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Carica il modello DINOv2
        from dinov2.models import load_model  # Assicurati di avere la libreria corretta
        dinov2_model = load_model("dinov2_vitb14")
        dinov2_model.eval()

        def extract_patch_feature(patch):
            """
            Data una patch in formato BGR, restituisce il vettore di feature estratto da DINOv2.
            """
            input_tensor = transform(patch).unsqueeze(0)  # forma (1, 3, 224, 224)
            with torch.no_grad():
                features = dinov2_model(input_tensor)  # Supponiamo restituisca un tensore (1, D)
            return features.cpu().numpy().flatten()

        # Funzione per calcolare statistiche sui logits in una patch
        def extract_logits_stats(logits_patch):
            """
            Data una patch della mappa dei logits, restituisce ad esempio la media e la varianza dei logits.
            Queste statistiche possono essere incluse nel vettore di feature.
            """
            mean_logit = np.mean(logits_patch)
            var_logit = np.var(logits_patch)
            return np.array([mean_logit, var_logit])

        # ------------------------------
        # STEP 4: Estrazione patch-wise e costruzione del dataset di feature arricchito
        # ------------------------------

        # Parametri della finestra mobile
        patch_size = 64  # Dimensione della patch (64x64 pixel)
        stride = 32  # Step della finestra mobile

        # Inizializza la mappa delle anomalie (stessa dimensione di roi_cropped_color)
        anomaly_map = np.zeros(roi_cropped_color.shape[:2], dtype=np.uint8)

        # Liste per raccogliere le feature e le posizioni delle patch
        patch_features = []
        patch_positions = []  # Per ricordare l'origine di ogni patch (coordinate)

        height, width, _ = roi_cropped_color.shape

        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                # Estrae la patch dell'immagine e quella dei logits
                patch = roi_cropped_color[i:i + patch_size, j:j + patch_size]
                logits_patch = logits_cropped[i:i + patch_size, j:j + patch_size]

                # Estrai il vettore di feature dalla patch usando DINOv2
                feature_vec = extract_patch_feature(patch)

                # Estrai le statistiche dai logits della patch
                logits_stats = extract_logits_stats(logits_patch)

                # Combina le feature estratte da DINOv2 e le statistiche sui logits
                # Ad esempio, concatenando i vettori (si assume che entrambe le parti siano normalizzate opportunamente)
                combined_feature = np.concatenate([feature_vec, logits_stats])

                patch_features.append(combined_feature)
                patch_positions.append((i, j))

        patch_features = np.array(patch_features)
        print("Dimensione del dataset di feature patch-wise:", patch_features.shape)

        # ------------------------------
        # STEP 5: Addestramento/Applicazione di Isolation Forest
        # ------------------------------

        # In un ambiente reale, il modello IF va addestrato su patch normali.
        # Qui, per esempio, addestriamo su tutte le patch estratte (presumendo che la maggior parte siano normali)
        if_model = IsolationForest(contamination=0.05, random_state=42)
        if_model.fit(patch_features)

        # Valutazione patch-wise e costruzione della mappa delle anomalie
        for idx, feat in enumerate(patch_features):
            prediction = if_model.predict(feat.reshape(1, -1))
            i, j = patch_positions[idx]
            if prediction[0] == -1:
                # Se la patch risulta anomala, aggiorniamo la mappa delle anomalie
                anomaly_map[i:i + patch_size, j:j + patch_size] = 255

        # Pulizia morfologica per eliminare piccoli artefatti (opzionale)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_map_clean = cv2.morphologyEx(anomaly_map, cv2.MORPH_OPEN, kernel)

        # ------------------------------
        # STEP 6: Visualizzazione dei risultati
        # ------------------------------

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Immagine Termica Originale")
        plt.imshow(image, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Maschera SAM (Binari)")
        plt.imshow(mask, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Segmentazione Anomalie (Con Logits)")
        plt.imshow(anomaly_map_clean, cmap='gray')
        plt.show()


def grounding_Dino_analyzer(image, model, caption, device, show=False, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25):
    print("Analysis with Grounding Dino")
    image_source, gd_image = load_image(image)

    gd_boxes, logits, phrases = predict(
        model=model,
        image=gd_image,
        caption=caption,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device,
    )
    annotated_frame = annotate(image_source=image_source, boxes=gd_boxes, logits=logits, phrases=phrases)

    if show:
        cv2.imshow("Visualizing results", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(phrases, logits)

    h, w, _ = image_source.shape
    gd_boxes = gd_boxes * torch.Tensor([w, h, w, h])
    gd_boxes = box_convert(boxes=gd_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return gd_boxes , phrases, logits.cpu().numpy()


def is_anomaly(mask, railway_mask, thermal_image=None):
    # Criteria 3: Location
    overlap = np.sum(np.logical_and(mask["segmentation"], railway_mask))
    if overlap / mask["area"] > 0.1 or mask["bbox"][1] > 320:  # <10% overlap with railway
        return True

    # Criteria 1: Stability and IoU
    if mask["stability_score"] > 0.8 or mask["predicted_iou"] > 0.8:
        return True

    # Criteria 2: Size
    if 50 < mask["area"] < 4000:
        return True

    # Criteria 4: Thermal intensity (if thermal data exists)
    if thermal_image is not None:
        mean_temp = np.mean(thermal_image[mask["segmentation"]])
        railway_temp = np.mean(thermal_image[railway_mask])
        if abs(mean_temp - railway_temp) > 5:  # 5°C deviation
            return True

    return False


def get_images(folder_path, max=None):
    for i, file in enumerate(os.listdir(folder_path)):
        if max is not None and i >= max:
            break
        image = Image.open(folder_path + "/" + file)
        image = np.array(image.convert("RGB"))
        thermal_image = (image - image.min()) / (image.max() - image.min()) * 255
        thermal_rgb = thermal_image.astype(np.uint8)
        yield thermal_rgb, file


def automatic_mask_generator(image, mask_generator, show=True, savefig=False, name="test"):
    print(f"Generating masks...")
    masks = mask_generator.generate(image)

    print(f"{len(masks)} maschere trovate.")
    print(masks[0].keys())

    anomalies = []
    for mask in masks:
        if 300 < mask['point_coords'][0][1] < 450:
            print(mask['point_coords'], mask['predicted_iou'], mask['stability_score'])
            anomalies.append(mask)
    if show:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(anomalies)
        plt.axis('off')
        if savefig:
            plt.savefig(f"./automask_results_3/{name[:-4]}.png")
        plt.show()
        print(f"---------------------------------")
    return anomalies


def create_grid(box, points_per_row=None):
    #proper creation of the inputs if wrong

    if points_per_row is None:
        points_per_row = [2, 2]

    rows = len(points_per_row)




    step_y = int(((box[3]-box[1]) / rows))
    points = []
    for row in range(rows):
        y = step_y * row + int(step_y / 2 ) + box[1]
        step_x = int(abs(box[2]-box[0]) /points_per_row[row])
        for i in range(points_per_row[row]):
            x = step_x * i + box[0] + int(step_x / 2)
            points.append([x, y])
    points = np.array(points)

    return points, np.ones(len(points))

def image_predictor(predictor, image, points, labels, folder_path, savefig=False, file_name=None):
    print("Predicting masks...")
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    plt.figure(figsize=(20, 20))
    plt.imshow(masks[0])
    plt.show()
    #show_masks(image, masks, scores, folder_path, point_coords=points, input_labels=labels, borders=True,
    #           savefig=savefig,
    #           file=file_name)
    print("-----------------------------")


def video_predictor(model_cfg_path, sam2_checkpoint, device, video_dir):
    print("Predicting masks...")
    predictor = build_sam2_video_predictor(model_cfg_path, sam2_checkpoint, device=device)

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]

    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    # points = np.array([[210, 350]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    # labels = np.array([1], np.int32)
    points = np.array([[107, 457],[250,250],[300, 200]])
    labels = np.ones(len(points))
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels
    )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }


    dpi = 100  # Risoluzione in punti per pollice
    width = 950 / dpi  # Larghezza in pollici
    height = 760 / dpi  # Altezza in pollici
    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        print(f"Rendering frame {out_frame_idx}...")
        plt.figure(figsize=(width, height), dpi=dpi)
        # plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            utility.show_mask_v(out_mask, plt.gca(), dpi, obj_id=out_obj_id, frame_name=out_frame_idx)
            # plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def is_mask_in_box(mask, box, margin=10):
    """
    Verifica se una maschera binaria è contenuta in un box, considerando un margine di tolleranza.

    Args:
        mask (np.ndarray): Maschera binaria di forma (1, H, W)
        box (list/tuple): Coordinate del box in formato [x1, y1, x2, y2]
        margin (int): Margine di tolleranza in pixel da aggiungere al box (default: 10)

    Returns:
        bool: True se la maschera è contenuta nel box allargato, False altrimenti
    """
    # Verifica input
    assert mask.shape[0] == 1, "La maschera deve avere shape (1, H, W)"
    assert len(box) == 4, "Il box deve avere 4 coordinate [x1, y1, x2, y2]"

    # Estrai coordinate del box e applica il margine
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - margin)  # Assicurati di non andare sotto 0
    y1 = max(0, y1 - margin)
    x2 = min(mask.shape[2], x2 + margin)  # Assicurati di non superare i limiti dell'immagine
    y2 = min(mask.shape[1], y2 + margin)

    # Trova le coordinate dei pixel non-zero nella maschera
    y_coords, x_coords = np.where(mask[0] > 0)

    if len(x_coords) == 0:  # Se la maschera è vuota
        return True

    # Verifica se tutti i punti della maschera sono dentro il box allargato
    mask_in_box = (
            (x_coords >= x1).all() and
            (x_coords <= x2).all() and
            (y_coords >= y1).all() and
            (y_coords <= y2).all()
    )

    return mask_in_box


def draw_mask(image, logits):  # Rename `mask` to `logits` for clarity
    """
    Plots a heatmap based on the magnitude of the output logits.
    """
    logits = logits.squeeze()  # Ensure float type
    logits[logits < 0] = 0  # Optional: Clip negatives (adjust as needed)

    # Apply sigmoid to convert logits to probabilities
    probabilities = 1 / (1 + np.exp(-logits))  # Now works with floats

    # Normalize to [0, 1] for visualization
    normalized = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))


    # Create heatmap
    cmap = matplotlib.colormaps['Reds']
    gradient = cmap(normalized, bytes=True)[:, :, :3]  # Drop alpha channel



    # Overlay heatmap on image
    masked_image = image.copy()
    masked_image = np.where(normalized[..., None] > 0, gradient, masked_image)

    return cv2.addWeighted(image, 0.6, masked_image, 0.4, 0)


def bin_otsu(out_mask_logits):
    # logits_mask_2d = np.squeeze(out_mask_logits)  # Shape: (H, W)
    #
    # # Step 1: Apply sigmoid to logits
    # sigmoid_mask = sigmoid(logits_mask_2d)  # Values are now in range [0, 1]
    #
    # # Step 2: Convert to uint8 (0-255 range for OpenCV)
    # sigmoid_mask_2d = (sigmoid_mask * 255).astype(np.uint8)  # Shape: (1, H, W)
    #
    # # Step 4: Apply Otsu's Thresholding
    # thresh_val, binary_mask = cv2.threshold(sigmoid_mask_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(thresh_val)
    # # thresh_val = 255 - thresh_val
    # # Step 5: Eliminate values below Otsu's threshold (set to 0)
    # binary_mask[sigmoid_mask_2d < thresh_val + (255 - thresh_val) * 0.42] = 0
    #
    # # Define the morphological kernel
    # opening_kernel = np.ones((5, 5), np.uint8)
    # closing_kernel = np.ones((10, 10), np.uint8)
    # #
    # # # Apply morphological opening & closing
    # mask_open = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_kernel)
    # mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
    # #
    # # # Convert back to (1, H, W)
    # mask_c = mask_closed / 255.0

    # out_mask_logits ha dimensione (1, 512, 640). Rimuoviamo la dimensione extra:
    mask_logits = np.squeeze(out_mask_logits, axis=0)  # ora mask_logits ha dimensione (512, 640)
    #mask_logits = np.maximum(mask_logits, 0)

    # Assicuriamoci che i valori negativi siano messi a 0
    probabilities = 1 / (1 + np.exp(-mask_logits))  # Now works with floats

    # Normalizziamo i valori nell'intervallo [0, 255]
    normalized =  (probabilities * 255).astype(np.uint8)
    #normalized_uint8 = normalized.astype(np.uint8)

    # Applichiamo la soglia di Otsu per ottenere una maschera binaria

    thresh_val, binary_mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask[normalized < thresh_val + (255 - thresh_val) * 0.70] = 0

    #Define the morphological kernel
    # opening_kernel = np.ones((5, 5), np.uint8)
    # closing_kernel = np.ones((5, 5), np.uint8)
    # #
    # # # Apply morphological opening & closing
    # mask_open = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_kernel)
    # mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, closing_kernel)


    # Applichiamo una colormap per ottenere una maschera a colori
    colored_mask = cv2.applyColorMap(mask_closed, cv2.COLORMAP_JET)

    # Se le dimensioni della colored_mask non corrispondono a quelle dell'immagine, ridimensioniamola
    if image.shape[:2] != binary_mask.shape[:2]:
        print(image.shape)
        print(binary_mask.shape)
        exit()

    masked_image = image.copy()
    co =  np.where(mask_closed[..., None] > thresh_val, colored_mask, masked_image)


    # Eseguiamo il blending: adesso image e colored_mask hanno lo stesso numero di canali e dimensioni
    overlay = cv2.addWeighted(image, 0.7, co, 0.3, 0)
    return overlay, co


def is_contained(box: np.ndarray, background_box: np.ndarray, threshold: float = 0.9) -> bool:
    """
    Verifica se il box è contenuto per almeno il 90% all'interno del background box.

    Args:
        box (np.ndarray): Array (x_min, y_min, x_max, y_max) del box da verificare.
        background_box (np.ndarray): Array (x_min, y_min, x_max, y_max) del box di sfondo.
        threshold (float): Percentuale minima di contenimento (default 0.9).

    Returns:
        bool: True se il box è contenuto almeno per il 90%, False altrimenti.
    """
    # Coordinate dei due box
    x_min_b, y_min_b, x_max_b, y_max_b = box
    x_min_bg, y_min_bg, x_max_bg, y_max_bg = background_box

    # Calcolo dell'intersezione tra i due box
    x_min_int = max(x_min_b, x_min_bg)
    y_min_int = max(y_min_b, y_min_bg)
    x_max_int = min(x_max_b, x_max_bg)
    y_max_int = min(y_max_b, y_max_bg)

    # Calcolo dell'area del box e dell'intersezione
    box_area = (x_max_b - x_min_b) * (y_max_b - y_min_b)
    inter_area = max(0, x_max_int - x_min_int) * max(0, y_max_int - y_min_int)
    if box_area > 30000:
        return False
    # Controllo la percentuale di contenimento
    return (inter_area / box_area) >= threshold if box_area > 0 else False


def segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Calcola IoU, Dice Coefficient, Precision e Recall tra la maschera predetta e la ground truth.

    Args:
        pred_mask (np.ndarray): Maschera predetta (binaria: 0 o 1)
        gt_mask (np.ndarray): Maschera ground truth (binaria: 0 o 1)

    Returns:
        dict: Dizionario con i valori delle metriche {"IoU": float, "Dice": float, "Precision": float, "Recall": float}
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    IoU = intersection / union if union > 0 else 0.0
    Dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0

    TP = intersection  # Veri positivi
    FP = np.logical_and(pred_mask, ~gt_mask).sum()  # Falsi positivi
    FN = np.logical_and(~pred_mask, gt_mask).sum()  # Falsi negativi

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return {"IoU": IoU, "Dice": Dice, "Precision": Precision, "Recall": Recall}

def reverse_frames(input_folder, output_folder):
    """
    Reads all JPEG frames from input_folder, reverses their order, and writes them to output_folder
    with filenames reflecting the new order.

    Parameters:
    - input_folder: path to the folder containing the original frames (e.g., "00000.jpg", "00001.jpg", etc.)
    - output_folder: path to the folder where reversed frames will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all JPEG files (modify the extension if necessary)
    frame_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

    if not frame_files:
        print("No .jpg files found in the input folder.")
        return

    # Sort files based on filename (assuming filenames are zero-padded numbers)
    frame_files = sorted(frame_files)

    # Reverse the list of frames
    reversed_frames = frame_files[::-1]

    # Determine the number of digits from the first filename (e.g., "00000" has 5 digits)
    num_digits = len(os.path.splitext(frame_files[0])[0])

    # Loop over reversed frames and copy them with new names
    for new_index, original_filename in enumerate(reversed_frames):
        # Create new filename with zero-padding
        new_filename = str(new_index).zfill(num_digits) + ".jpg"

        # Build full paths for input and output files
        src_path = os.path.join(input_folder, original_filename)
        dst_path = os.path.join(output_folder, new_filename)

        # Copy the file to the new location with the new name
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} -> {dst_path}")



def find_anomaly(image, logits):


    # contorni


    # logits bassi



    # image shape (3, H, W)
    # logits shape (1, H, W)

    l = np.squeeze(logits) # Shape: (H, W)
    l[l < 0] = 0

    # Step 1: Apply sigmoid to logits
    sigmoid_mask = sigmoid(l)  # Values are now in range [0, 1]

    # Step 2: Convert to uint8 (0-255 range for OpenCV)
    sigmoid_mask_2d = (sigmoid_mask * 255).astype(np.uint8)  # Shape: (1, H, W)
    sig = (sigmoid_mask_2d > 127).astype(np.uint8)

    # Step 3: find if there's holes in the mask
    coord_buchi, labels_ = find_holes(sig, 400)
    if len(coord_buchi) > 0:
        print("buchi", coord_buchi)
        print("buchi_l", l)

        m, s, _ = rd.image_predictor.predict(
            point_coords=coord_buchi,
            point_labels=labels_,
            multimask_output=False,
        )
        utility.show_masks(image, m, s, point_coords=coord_buchi, input_labels=l)

    ## logits bassi

    # Step 4: Apply Otsu's Thresholding
    thresh_val, binary_mask = cv2.threshold(sigmoid_mask_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(thresh_val)

    # Step 5: Eliminate values below Otsu's threshold (set to 0)
    # binary_mask[ thresh_val + (255-thresh_val)*0.42 > sigmoid_mask_2d > 0 ] = 0

    binary_mask = binary_mask.astype(np.uint8)
    # otteniamo i buchi
    # Creare una copia della maschera per il flood fill
    mask_floodfill = binary_mask.copy()

    # Creare una maschera extra per il flood fill (deve essere 2 pixel più grande)
    h, w = binary_mask.shape
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Eseguire il flood fill dallo sfondo esterno (pixel in alto a sinistra)
    cv2.floodFill(mask_floodfill, floodfill_mask, (0, 0), 255)

    # Invertire la flood fill per ottenere solo i buchi
    holes_mask = cv2.bitwise_not(mask_floodfill) & ~binary_mask

    # Riportare il risultato a float64 con valori in [0,1]
    holes_mask_float64 = (holes_mask / 255).astype(np.float64)

    # Define the morphological kernel
    opening_kernel = np.ones((6, 6), np.uint8)
    closing_kernel = np.ones((6, 6), np.uint8)

    # # Apply morphological opening & closing
    # mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, closing_kernel)
    # mask_open = cv2.morphologyEx(holes_mask_float64, cv2.MORPH_OPEN, opening_kernel)
    #
    # mask_closed = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, closing_kernel)

    # Convert back to (1, H, W)
    #mask_final = np.expand_dims(binary_mask, axis=0)  # Shape: (1, H, W)

    print("max_sig", np.max(sigmoid_mask_2d))
    print("min_sig", np.min(sigmoid_mask_2d))
    print("max_binary", np.max(binary_mask))
    print("min_binary", np.min(binary_mask))
    sig = (sigmoid_mask_2d > 127).astype(np.uint8)
    bin = (binary_mask > 0).astype(np.uint8)
    # binary_mask = binary_mask / 255.0
    mask_final = np.logical_and(sig, np.logical_not(bin)).astype(np.uint8)

    print("max_fin", np.max(mask_final))
    print("min_fin", np.min(mask_final))
    print("mask_final shape", mask_final.shape)
    mask_final = np.expand_dims(mask_final, axis=0)  # Shape: (1, H, W)

    utility.show_masks(image, mask_final, np.array([0]), borders=False)
    #Search for holes

def traditional_detection(image):
    """Metodo basato su tecniche tradizionali di computer vision"""
    # Applica filtro gaussiano per ridurre il rumore
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Calcola il gradiente
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Sogliatura adattiva
    threshold = cv2.adaptiveThreshold(
        gradient_magnitude.astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Trova i contorni
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra i contorni per dimensione
    min_area = 100  # area minima in pixel
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Genera bounding box
    boxes = []
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if area < 200:  # area minima
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])

    return np.array(boxes)


def find_holes(mask, min_hole_size=50):
    """
    Trova i buchi all'interno di un'area nella maschera binaria e restituisce i centroidi dei buchi abbastanza grandi.

    Args:
        mask (np.ndarray): Maschera binaria (1, H, W) di tipo uint8 (valori 0 e 1 o 0 e 255).
        min_hole_size (int): Dimensione minima per considerare un buco valido.

    Returns:
        List[Tuple[int, int]]: Lista di centroidi dei buchi (x, y).
    """
    # Rimuove la prima dimensione se la maschera è (1, H, W) -> diventa (H, W)
    #mask = mask.squeeze()

    # Trova i contorni degli oggetti principali
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crea una maschera nera di sfondo per disegnare gli oggetti trovati
    filled_mask = np.zeros_like(mask)

    # Riempie completamente gli oggetti principali per ottenere solo i buchi
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Trova i buchi confrontando la maschera riempita con l'originale
    holes_mask = np.logical_and(filled_mask > 0, mask == 0).astype(np.uint8) * 255

    # Trova i contorni dei buchi
    hole_contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in hole_contours:
        area = cv2.contourArea(cnt)
        if area > min_hole_size:  # Filtra i buchi troppo piccoli
            M = cv2.moments(cnt)
            if M["m00"] > 0:  # Evita divisioni per zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append([cx, cy])

    return np.array(centroids),np.random.choice(np.arange(2, 51), size=len(centroids), replace=False)


def get_centers_from_mask(mask, min_area=100):
    """
    Data una maschera, ritorna le coordinate (centroide) delle aree con area maggiore di min_area.

    Args:
        mask (np.ndarray): Immagine in scala di grigi o binaria.
                          Gli oggetti di interesse dovrebbero avere valori > 0.
        min_area (int): Soglia minima dell'area (in pixel) per considerare la regione.

    Returns:
        np.ndarray: Array di coordinate (x, y) dei centroidi delle aree rilevate.
                   Shape: (N, 2) dove N è il numero di aree trovate.
    """
    # Controllo input
    if mask is None or mask.size == 0:
        return np.array([])

    # Assicuriamoci che la maschera sia 2D
    if mask.ndim > 2:
        mask = np.squeeze(mask)


    # Convertiamo a uint8 se necessario
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Binarizziamo la maschera
    #_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Trova le componenti connesse
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    centers = []
    # Il label 0 è il background, partiamo da 1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_area:
            center = [int(centroids[i][0]), int(centroids[i][1])]
            centers.append(center)

    return np.array(centers)


def find_corresponding_segmentation(image_path, segmentation_folder):
    """
    Trova il file di segmentazione corrispondente a un'immagine di input.

    Args:
        image_path (str): Percorso del file immagine originale.
        segmentation_folder (str): Directory contenente le segmentazioni.

    Returns:
        str | None: Percorso del file di segmentazione corrispondente, se esiste. Altrimenti None.
    """
    # Estrai il numero dal nome del file originale
    filename = os.path.basename(image_path)  # Ottieni solo il nome del file
    number_part = filename.split('_')[-1].split('.')[0]  # Ottieni la parte numerica

    # Costruisci il nome del file di segmentazione
    segmentation_filename = f"segmentazione_oggetti_{number_part}.png"
    segmentation_path = os.path.join(segmentation_folder, segmentation_filename)

    # Controlla se il file di segmentazione esiste
    if os.path.exists(segmentation_path):
        return segmentation_path
    else:
        return None  # Se il file non esiste







if __name__ == '__main__':

    ## SETTING THE MODELS

    ## SAM2

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")



    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    # loading sam paramters
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg_path = "configs/sam2.1/sam2.1_hiera_t.yaml"

    np.random.seed(3)


    ##  Grounding DINO

    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth",device)
    #IMAGE_PATH = "inserimento_oggetti/inserimento_oggetti_00064.jpg"
    TEXT_PROMPT = "object ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25





    # Where to retrive images
    folder_path = "inserimento_oggetti"



    # creating the background box
    background_box = np.array([0, 256, 640, 512])


    rd = RailAnomalyDetector(model_cfg_path, sam2_checkpoint, device=device)


    # Setting statistics
    gd_iou, h_iou, otsu_iou = 0, 0, 0
    gd_dice , h_dice, otsu_dice = 0, 0, 0
    gd_precision , h_precision, otsu_precision = 0, 0, 0
    gd_recall, h_recall, otsu_recall = 0, 0, 0

    n_frames = 0
    skipped = 0


    # Extract frames
    frame_names = [
        p for p in os.listdir(folder_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # Sorting frames
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    # FOR VIDEO ANALYSIS
    #rd.video_analysis('s_c_f','seg_new_video', model)

    # IMAGE ANALYSIS
    try:
        for p in frame_names:


            print('----------------------')
            image =  cv2.imread(os.path.join(folder_path, p))
            ground_truth = cv2.imread(find_corresponding_segmentation(p,
                                                    'segmentazione_oggetti'),
                                      cv2.IMREAD_GRAYSCALE)

            if cv2.countNonZero(ground_truth) < 170:
                print("oggetto troppo piccolo")
                skipped += 1
                continue

            if n_frames % 20 == 0:
                print("frame saltati: ", skipped)



            name = str(p)
            print("Analyzing image:", name)
            print()
            rd.image_predictor.set_image(image)

            m_gd, m_h, m_l = None, None, None
            n_frames += 1

            # ## Grounding DINO - Finding objects in the bounding box
            image_source, gd_image = load_image(folder_path+'/'+name)

            gd_boxes, logits, phrases = predict(
                model=model,
                image=gd_image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device=device,
            )
            # annotated_frame = annotate(image_source=image_source, boxes=gd_boxes, logits=logits,
            #                            phrases=phrases)
            # cv2.imwrite("debug/Grounding_Dino/" + name, annotated_frame)

            h, w, _ = image_source.shape
            gd_boxes = gd_boxes * torch.Tensor([w, h, w, h])
            gd_boxes = box_convert(boxes=gd_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            for box in gd_boxes:
                if is_contained(box, background_box,1):
                    m_gd, s_gd, _ = rd.image_predictor.predict(
                        box=box,
                        multimask_output=False,
                    )
                    # salvataggio dell'immagine
                    utility.show_masks(image, m_gd, s_gd,borders=False,show=False,savefig=True,save_path="debug/gd2/",
                                       save_name=name,box_coords=box)


                    d_gd = segmentation_metrics(m_gd, ground_truth)
                    for metric in d_gd:
                        if metric == 'IoU':
                            gd_iou += d_gd[metric]
                        elif metric == 'Dice':
                            gd_dice += d_gd[metric]
                        elif metric == 'Precision':
                            gd_precision += d_gd[metric]
                        elif metric == 'Recall':
                            gd_recall += d_gd[metric]

            print("Grounding Dino - frame: ",n_frames, ", IoU: ", gd_iou/n_frames, ", Dice: ", gd_dice/n_frames,
                  ", Precision: ", gd_precision/n_frames, ", Recall: ", gd_recall/n_frames)






            # GRID of points

            points ,labels = create_grid(background_box, points_per_row=[3, 4, 5])

            masks, scores, logits = rd.image_predictor.predict(
                point_coords=points,
                point_labels=labels,
                #box=background_box,
                multimask_output=False,
                return_logits=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            #print(masks.shape) # (3, 512, 640)
            utility.show_masks(image,masks>0,scores,input_labels=labels,point_coords=points,show=False,savefig=True,
                               save_path='debug/grid/',save_name=p)



            l = np.squeeze(masks)
            l[l < 0] = 0
            # for i in range(4):
            #
            #     mask, score, _ = rd.image_predictor.predict(
            #         point_coords=points[i:i+5],
            #         point_labels=labels[i:i+5],
            #         box=background_box,
            #         return_logits=True,
            #         multimask_output=False,
            #     )
            #     log += mask

            # boxes = traditional_detection(im)
            # for box in boxes:
            #
            #     m, s, _ = rd.image_predictor.predict(
            #         box=box,
            #         multimask_output=False
            #     )
            #     utility.show_masks(image, m, s, box_coords=box)
            #     input("Press Enter to continue...")
            #



            #utility.show_masks(image, mask,score,box_coords=background_box)
            # seg_im = draw_mask(image, mask)
            # cv2.imshow(name, seg_im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # Step 3: Ensure it's in 2D format (H, W) for OpenCV
            #logits_mask_2d = np.squeeze(log)  # Shape: (H, W)

            sigmoid_mask = sigmoid(l)  # Values are now in range [0, 1]

            # Step 2: Convert to uint8 (0-255 range for OpenCV)
            sigmoid_mask_2d = (sigmoid_mask * 255).astype(np.uint8)  # Shape: (1, H, W)


            sig = (sigmoid_mask_2d > 127).astype(np.uint8) # binarify the mask to search for holes

            coord_holes, holes_labels = find_holes(sig, 100)
            if len(coord_holes)>0:

                m_h,s_h,_ = rd.image_predictor.predict(
                    point_coords=coord_holes,
                    point_labels=holes_labels,
                    multimask_output=False,
                )
                utility.show_masks(image, m_h, s_h, borders=False, show=False, savefig=True, point_coords=coord_holes,
                                   input_labels=holes_labels, save_path='debug/holes/', save_name=name)

                d_h = segmentation_metrics(m_h, ground_truth)
                for metric in d_h:
                    if metric == 'IoU':
                        h_iou += d_h[metric]
                    elif metric == 'Dice':
                        h_dice += d_h[metric]
                    elif metric == 'Precision':
                        h_precision += d_h[metric]
                    elif metric == 'Recall':
                        h_recall += d_h[metric]

            print("Holes - frame: ", n_frames, ", IoU: ", h_iou / n_frames, ", Dice: ", h_dice / n_frames,
                  ", Precision: ", h_precision / n_frames, ", Recall: ", h_recall / n_frames)

            # Binarify with otsu

            thresh_val, binary_mask = cv2.threshold(sigmoid_mask_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            binary_mask[binary_mask > 127] = 1


            if binary_mask.dtype != np.uint8:
                binary_mask = binary_mask.astype(np.uint8)


            coord_holes, o_labels = find_holes(binary_mask, 100)
            if len(coord_holes) > 0:
                m_o, s_o, _ = rd.image_predictor.predict(
                    point_coords=coord_holes,
                    point_labels=o_labels,
                    multimask_output=False,
                )
                utility.show_masks(image, m_o, np.array([0]), borders=False, point_coords=coord_holes, input_labels=o_labels,
                                   show=False, savefig=True, save_path='debug/otzu/', save_name=name)

                d_o = segmentation_metrics(m_o, ground_truth)
                for metric in d_o:
                    if metric == 'IoU':
                        otsu_iou += d_o[metric]
                    elif metric == 'Dice':
                        otsu_dice += d_o[metric]
                    elif metric == 'Precision':
                        otsu_precision += d_o[metric]
                    elif metric == 'Recall':
                        otsu_recall += d_o[metric]


            print("Otsu - frame: ", n_frames, ", IoU: ", otsu_iou / n_frames, ", Dice: ", otsu_dice / n_frames,
                  ", Precision: ", otsu_precision / n_frames, ", Recall: ", otsu_recall / n_frames)
    except(KeyboardInterrupt, SystemExit):
        print("Exiting...")
        print('iou', gd_iou, h_iou, otsu_iou)
        print('dice', gd_dice, h_dice, otsu_dice)
        print('precision', gd_precision, h_precision, otsu_precision)
        print('recall', gd_recall, h_recall, otsu_recall)
        print('n_frames ', n_frames)
        print('skipped ',skipped)
