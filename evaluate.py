

def evaluate_model(model, val_loader, ind2cat, ind2vad):
    model.eval()
    cat_preds, cat_labels = [], []
    cont_preds, cont_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            context, body, clip_context, bbox_body, bbox_face, scenario = [batch[k].to(device) for k in ['context', 'body', 'clip_context', 'bbox_body', 'bbox_face', 'scenario']]
            out_cat, out_cont = model(context, body, clip_context, bbox_body, bbox_face, scenario)

            cat_preds.append(out_cat.cpu().numpy())
            cont_preds.append(out_cont.cpu().numpy())
            cat_labels.append(batch['cat_label'].cpu().numpy())
            cont_labels.append(batch['cont_label'].cpu().numpy())

    cat_preds = np.concatenate(cat_preds, axis=0)
    cat_labels = np.concatenate(cat_labels, axis=0)
    cont_preds = np.concatenate(cont_preds, axis=0)
    cont_labels = np.concatenate(cont_labels, axis=0)

    # 결과 출력
    ap = test_ap(cat_preds, cat_labels, ind2cat)
    vad = test_vad(cont_preds, cont_labels, ind2vad)

    # 결과를 직접 출력
    print("Average Precision per category: ", ap)
    print("Mean AP: ", np.mean(ap))
    print("VAD Errors: ", vad)
    print("Mean VAD Error: ", np.mean(vad))

ind2cat = val_dataset.ind2cat
ind2vad = val_dataset.ind2vad
model.load_state_dict(torch.load('/content/drive/MyDrive/CLIPBLIP/log/model_epoch29_iter300.pth'))

evaluate_model(model, val_loader, ind2cat, ind2vad)

