from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./bcbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

# error_text_n = """E1: Multiple attempts — performing multiple piercings without regrasping the needle in tissue.
# E2: Needle drop or slip while the needle is still in tissue.
# E3: Instrument goes out of view.
# E4: Needle goes out of view.
# E5: Tissue damage or poor tissue stabilization.
# E6: Incorrect angle when grasping the needle (not perpendicular).
# E7: Incorrect grasping position along the needle (normally second or third position).
# E8: Applying excessive force.
# E9: Needle does not follow its natural curvature.
# E10: Incorrect needle entry angle (not perpendicular).
# E11: Grasping the needle tip.
# E12: Suture loosened after tying.
# E13: Thread caught in the instrument.
# E14: Knot is not square (C or reverse C shape).
# E15: Inadequate number of throws in the knot.
# E16: Suture pulled through tissue before tying the knot.
# E17: Incorrect spacing between needle drives (too close or too far).
# E18: Suture not pulled through between needle drives.
# E19: Suture entanglement.
# E20: Fraying of the suture.
# E21: Snapping or breaking the suture.
# E22: Unsafe or incorrect needle disposal.
# E23: Poor camera control (blurred or incorrect view).
# E24: Poor instrument control (clashing, improper use of third arm, or non-dominant hand)."""

error_text_p = """	1.	Only one piercing per needle regrasp; no repeated piercings.
	2.	Needle handled securely; no drop or slip in tissue.
	3.	Instruments remain fully visible at all times.
	4.	Needle remains fully visible in the camera view.
	5.	Tissue handled carefully; stabilized properly, no damage.
	6.	Needle grasped at correct perpendicular angle.
	7.	Needle grasped at correct position along shaft.
	8.	Applied force is appropriate; no excessive pressure.
	9.	Needle follows its natural curvature smoothly.
	10.	Needle entry angle is correct and perpendicular.
	11.	Needle tip not grasped; proper mid-shaft grasp used.
	12.	Suture remains tight and secure after tying.
	13.	Thread is free; not caught in instruments.
	14.	Knot is square and properly formed.
	15.	Correct number of throws used for secure knot.
	16.	Suture is pulled through tissue only after knot is tied.
	17.	Needle drives spaced correctly; neither too close nor too far.
	18.	Suture correctly pulled through between needle drives.
	19.	Suture is free from entanglement.
	20.	Suture maintains integrity; no fraying occurs.
	21.	Suture remains intact; no snapping or breaking.
	22.	Needle disposed safely and correctly.
	23.	Camera control is optimal; view is clear and correct.
	24.	Instrument control is precise; no clashing, proper use of all arms."""

# encode
# inputs_n = tokenizer(error_text_n, padding=True, truncation=True, return_tensors="pt").to(device)
inputs_p = tokenizer(error_text_p, padding=True, truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    # outputs_n = model(**inputs_n)
    # text_feature_n = outputs_n.last_hidden_state.mean(dim=1)  # 平均池化

    outputs_p = model(**inputs_p)
    text_feature_p = outputs_p.last_hidden_state.mean(dim=1)  # 平均池化

# torch.save((text_feature_n), "./data/error_text_emb_n.pt")
torch.save((text_feature_p), "./data/error_text_emb_p.pt")
print("Saved error_text_emb.pt with shapes:", text_feature_p.shape)
