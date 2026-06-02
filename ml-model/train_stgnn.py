import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from graph.build_graph import build_road_graph
from graph.temporal_graph import create_temporal_graphs
from models.stgnn import STGNN

def main():
    print("🚀 train_stgnn.py started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("📥 Loading & building graph...")
    X, edge_index = build_road_graph("data/US_Accidents.csv")

    print("⏳ Creating temporal graphs...")
    graphs = create_temporal_graphs(X, edge_index, window=5)
    print(f"📊 Total temporal graphs: {len(graphs)}")

    model = STGNN(in_channels=X.shape[1], hidden_channels=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    print("🚀 Training started...")
    model.train()
        # ---------- Evaluation ----------
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            pred = model(g.x, g.edge_index)
            y_true.append(int(g.y.item()))
            y_pred.append(int(pred.item() > 0.5))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n📊 Evaluation Metrics")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    for epoch in range(1, 6):
        total_loss = 0

        for g in graphs:
            optimizer.zero_grad()

            g = g.to(device)
            pred = model(g.x, g.edge_index)

            loss = loss_fn(pred.unsqueeze(0), g.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / len(graphs):.4f}")

    print("✅ Training completed successfully")

    # Save the trained model while 'model' is still in scope
    torch.save(model.state_dict(), "stgnn_model.pth")
    print("💾 Model saved as stgnn_model.pth")

if __name__ == "__main__":
    main()
