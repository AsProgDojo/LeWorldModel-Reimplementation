import torch

from lewm.smoke import (
    ACTION_DIM,
    HISTORY_SIZE,
    IMAGE_SIZE,
    NUM_PREDS,
    make_fake_batches,
    run_smoke,
    shuffle_actions,
)

def test_fake_batches_have_the_expected_shapes_and_are_reproducible():
    first = make_fake_batches(num_batches=2, batch_size=3, seed=7)
    again = make_fake_batches(num_batches=2, batch_size=3, seed=7)
    other = make_fake_batches(num_batches=2, batch_size=3, seed=8)

    frames = HISTORY_SIZE + NUM_PREDS
    assert len(first) == 2
    assert first[0]["pixels"].shape == (3, frames, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert first[0]["action"].shape == (3, frames, ACTION_DIM)
    assert torch.equal(first[0]["pixels"], again[0]["pixels"])
    assert not torch.equal(first[0]["pixels"], other[0]["pixels"])

def test_fake_frames_show_a_square_that_moves_with_the_action():
    batch = make_fake_batches(num_batches=1, batch_size=4, seed=0)[0]
    pixels, action = batch["pixels"], batch["action"]

    assert set(pixels.unique().tolist()) <= {0.0, 1.0}
    assert (pixels.flatten(2).sum(-1) > 0).all()  # a square is visible in every frame

    # centre of mass of each frame, in (x, y)
    coords = torch.arange(IMAGE_SIZE, dtype=torch.float32)
    mask = pixels[:, :, 0]
    total = mask.sum(dim=(-1, -2))
    centre_x = (mask.sum(-2) * coords).sum(-1) / total
    centre_y = (mask.sum(-1) * coords).sum(-1) / total
    moved = torch.stack([centre_x, centre_y], dim=-1).diff(dim=1)  # (B, T-1, 2)

    # the square moves in the direction of the action, unless it is stopped by the border
    inside = (moved.abs() > 0.5).all(-1)
    assert inside.any()
    assert (torch.sign(moved[inside]) == torch.sign(action[:, :-1][inside])).all()


def test_shuffle_actions_changes_only_the_actions():
    batches = make_fake_batches(num_batches=1, batch_size=4, seed=0)
    shuffled = shuffle_actions(batches)

    assert torch.equal(shuffled[0]["pixels"], batches[0]["pixels"])
    assert torch.equal(shuffled[0]["action"][1], batches[0]["action"][0])


def test_smoke_run_learns_without_collapsing(tmp_path):
    results = run_smoke(tmp_path, device="cpu", precision="32", num_epochs=3)

    # the loss goes down
    assert results["last_loss"] < 0.7 * results["first_loss"]
    assert results["last_pred_loss"] < 0.7 * results["first_pred_loss"]
    # no collapse: SIGReg keeps the embeddings spread out, close to unit scale
    assert 0.3 < results["embedding_std_mean"] < 3.0
    assert results["embedding_std_min"] > 0.05
    # the predictor really uses the actions (and therefore the embeddings carry the position)
    assert results["val_pred_loss"] < 0.8 * results["val_pred_loss_shuffled_actions"]
