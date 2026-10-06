import numpy as np
import torch


@torch.no_grad()
def sliding_window_predict(
    model,
    image,
    device,
    patch_size=48,
    stride=48,
):

    model.eval()

    h, w = image.shape

    probability = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    count = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    ys = list(
        range(
            0,
            max(1, h - patch_size + 1),
            stride,
        )
    )

    xs = list(
        range(
            0,
            max(1, w - patch_size + 1),
            stride,
        )
    )

    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)

    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)

    for y in ys:

        for x in xs:

            patch = image[
                y:y + patch_size,
                x:x + patch_size,
            ]

            tensor = (
                torch.from_numpy(patch)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            logits = model(tensor)

            pred = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()

            probability[
                y:y + patch_size,
                x:x + patch_size,
            ] += pred

            count[
                y:y + patch_size,
                x:x + patch_size,
            ] += 1

    probability /= np.maximum(
        count,
        1e-8,
    )

    return probability