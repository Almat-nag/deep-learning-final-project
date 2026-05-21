with open("results/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
model.load_state_dict(torch.load("results/final_lstm_model.pt", map_location=device))
