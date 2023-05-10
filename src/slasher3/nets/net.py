class Net:
    net = None
    name : str
    folder : str
    device = None
    training_loader = None
    validation_loader = None
    epochs : int
    training_lr : float
    patience : int
    output_width : int
    optimizer = None
    loss_function = None
    score_function = None
    results = None

    def __init__(self, name, folder, device, output_width, training_loader=None, validation_loader=None, epochs=None, training_lr=None, patience=None, net_save=None):
        self.name = name
        self.net_folder = folder
        self.device = device
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.training_lr = training_lr
        self.patience = patience
        self.output_width = output_width
        if net_save: # load from disk
            self.from_file(net_save)
        else:
            self.results = self.fit()
    
    def fit(self):
        pass

    def step(self):
        pass

    def evaluate(self):
        pass

    def mask_and_save(self):
        pass

    def from_file(self, net_save : str):
        pass