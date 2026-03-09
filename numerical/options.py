import torch

class Options:
    def __init__(self, model_name, isTrain=True):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========================
        # Normalization / de-normalization helpers
        # =========================
        # If your training CSV is already normalized to [0,1], you can keep
        # normalize="none".
        # If your CSV is raw (original units), set normalize="minmax" and the
        # loader will normalize to [0,1] and save min/max stats so sampling can
        # automatically de-normalize back to original units.
        self.normalize = "none"       # "none" or "minmax"
        self.norm_eps = 1e-8
        self.norm_stats_path = None   # e.g. "/path/to/output_model/norm_stats.npz"
        self.raw_data_file_for_stats = None  # optional raw CSV used only to compute min/max
        # Load min/max stats even if you are NOT normalizing inside this code.
        # (Useful if your CSV is already normalized but you still want to
        # automatically de-normalize generated samples.)
        self.load_norm_stats = True

        # Sampling output control
        # - "sigmoid" keeps generated normalized values in (0,1) smoothly
        # - "none" returns raw diffusion output (may go out of range)
        self.sample_activation = "sigmoid"  # "sigmoid" or "none"

        # Whether sample() should also return de-normalized outputs (if stats exist)
        self.sample_return_denorm = True
        self.n_epochs = 200
        self.level = "No"  # ***
        self.seq_len = 32  # station——288, driver——720
        self.cond_flag = "conditional"  # ***
        # if isTrain:
        #     self.batch_size = 4  # station——4, driver——8
        #     self.shuffle = True
        # else:
        #     self.batch_size = 1
        #     self.shuffle = False
        if model_name == "diffusion":
            self.init_lr = 1e-3
            self.network = "attention"  # "attention" or "cnn"
            self.input_dim = 20
            self.hidden_dim = 256
            self.cond_dim = 4
            self.nhead = 8
            self.beta_start = 1e-4
            self.beta_end = 0.02
            self.n_steps = 100
            self.schedule = "linear"  # "linear"
        elif model_name == "gan":
            self.lr_G = 1e-5
            self.lr_D = 1e-3
            self.network = "attention"  # "attention" or "cnn"
            self.input_size = 18
            self.latent_size = 48
            self.condition_size = 16
            self.hidden_G = 64
            self.hidden_D = 32
            self.beta_end = 0.5
            self.n_steps = 50
            self.seq_len = 18
            self.schedule = "quadratic"  # "linear"