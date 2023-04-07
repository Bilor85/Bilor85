import numpy as np
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from model.loss import PortfolioTransformerLoss
from evaluate import evaluate_inference_mode
from utils.tools import bulk_log
from utils.model import get_loggers, get_model, get_num_param, save_model, get_dataloader
from utils.tools import get_config, write_message
from utils.tools import to_device
from evaluate import evaluate
from utils.metrics import PortfolioMetrics
# from utils.metrics import metrics_preproc
from utils.tools import calculate_port_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    # Get dataloader
    train_dataset, train_loader = get_dataloader(configs, train=True)
    print("Train dataset length:", len(train_dataset))

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    num_param = get_num_param(model)


    pt_loss = PortfolioTransformerLoss(configs).to(device)
    print("Number of PortfolioTransformer Parameters:", num_param)

    # Init loggers
    train_logger, train_log_path, val_logger, val_log_path = get_loggers(configs)
    write_message(train_log_path, str(configs))

    # Metric loggers
    portfolio_metrics = PortfolioMetrics( config)

    # Training
    step = args.restore_step if args.restore_step == 0 else args.restore_step + 1
    epoch = 1
    grad_acc_step = configs["optimizer"]["grad_acc_step"]
    grad_clip_thresh = configs["optimizer"]["grad_clip_thresh"]
    total_step = configs["training"]["steps"]["total_step"]
    log_step = configs["training"]["steps"]["log_step"]
    save_step = configs["training"]["steps"]["save_step"]
    val_step = configs["training"]["steps"]["val_step"]
    n = configs["model"]["n_assets"]
    tau = configs["model"]["n_window_timesteps"]
    transaction_cost_rate = configs["training"]["loss"]["transaction_cost_rate"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(train_loader), desc="Epoch {}".format(epoch), position=1)
        for batches in train_loader:
            for batch in batches:
                
                batch = to_device(batch, device)
               
                # Forward
                output = model(*batch[2:-1])
                
                # Call Loss
                losses = pt_loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    grad_norms = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f},".format(*losses)
                    
                    write_message(train_log_path, message1 + message2 + "\n")
                    outer_bar.write(message1 + message2)

                    bulk_log(train_logger, portfolio_metrics, transaction_cost_rate, step, batch, output, grad_norms,
                             losses, optimizer, n, tau)

                if step % val_step == 0:
                    print("Validation...")
                    model.eval()
                    message = evaluate(model, step, configs, val_logger)
                    write_message(val_log_path, message)

                    evaluate_inference_mode(model, step, configs, portfolio_metrics, val_logger)

                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    save_model(model, optimizer, configs, step)

                if step == total_step:
                    save_model(model, optimizer, configs, step)
                    quit()

                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to config.json"
    )
    args = parser.parse_args()

    # Read Config
    config = get_config(args.config)

    main(args, config)