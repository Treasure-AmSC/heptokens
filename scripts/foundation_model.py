
    direct_hist = None
    direct_res = None
    if run_direct:
        log.info("\n%s", "=" * 60)
        log.info("A) DIRECT CLASSIFIER (from scratch)")
        log.info("%s", "=" * 60)
        direct_model = SequenceClassifier(SequenceBackbone(config))
        direct_hist = trainer.train_classifier(
            direct_model,
            train_cls,
            val_cls,
            epochs=args.direct_epochs,
            lr=args.lr,
            stage_name="Direct classifier training from scratch",
            progress_name="Direct epoch",
        )
        direct_res = trainer.evaluate(direct_model, test_cls)
        log.info("  Direct: Acc=%.4f AUC=%.4f", direct_res["accuracy"], direct_res["auc"])
    else:
        log.info("Skipping direct classifier because --run=%s", args.run)

    pretrain_hist = None
    if not run_foundation:
        log.info("Skipping foundation model because --run=%s", args.run)
        foundation_backbone = None
    elif args.load_pretrained:
        log.info("Loading pre-trained backbone from %s", args.load_pretrained)
        foundation_backbone = SequenceBackbone(config)
        foundation_backbone.load_state_dict(
            torch.load(args.load_pretrained, weights_only=False, map_location="cpu")
        )
    elif args.pretrain_parquet:
        log.info("Loading pre-training data...")
        train_pt, val_pt = make_pretrain_loaders(
            args.pretrain_parquet,
            batch_size=args.batch_size,
            max_sequences=args.max_pretrain_sequences,
            token_column=args.token_column,
            mask_column=args.mask_column,
            type_column=optional_column(args.type_column),
        )
        log.info("  Pre-train: %s, Val: %s", len(train_pt.dataset), len(val_pt.dataset))
        masked_model = MaskedSequenceModel(config)
        pretrain_hist = trainer.pretrain(
            masked_model,
            train_pt,
            val_pt,
            epochs=args.pretrain_epochs,
            lr=args.lr,
        )
        foundation_backbone = masked_model.backbone
        if args.save_pretrained:
            torch.save(foundation_backbone.state_dict(), args.save_pretrained)
            log.info("  Saved pre-trained backbone to %s", args.save_pretrained)
    else:
        log.warning("No pre-training data or checkpoint provided. Skipping foundation model.")
        foundation_backbone = None

    foundation_hist = None
    foundation_res = None
    if foundation_backbone is not None:
        log.info("\n%s", "=" * 60)
        log.info("B) FOUNDATION MODEL (pre-trained + fine-tuned)")
        log.info("%s", "=" * 60)
        foundation_model = SequenceClassifier(
            foundation_backbone,
            freeze_backbone=args.freeze_backbone,
        )
        foundation_hist = trainer.train_classifier(
            foundation_model,
            train_cls,
            val_cls,
            epochs=args.finetune_epochs,
            lr=args.lr,
            stage_name="Foundation classifier fine-tuning",
            progress_name="Foundation fine-tune epoch",
        )
        foundation_res = trainer.evaluate(foundation_model, test_cls)
        foundation_res["pretrain_hist"] = pretrain_hist

    if direct_res is not None and foundation_res is not None:
        delta_acc = foundation_res["accuracy"] - direct_res["accuracy"]
        delta_auc = foundation_res["auc"] - direct_res["auc"]
        log.info("\n%s", "=" * 60)
        log.info("COMPARISON")
        log.info("%s", "=" * 60)
        log.info("  Direct:     Acc=%.4f AUC=%.4f", direct_res["accuracy"], direct_res["auc"])
        log.info(
            "  Foundation: Acc=%.4f AUC=%.4f",
            foundation_res["accuracy"],
            foundation_res["auc"],
        )
        log.info("  Delta:      Acc=%+.4f AUC=%+.4f", delta_acc, delta_auc)
        plot_comparison(direct_res, foundation_res, direct_hist, foundation_hist, output_dir)
    else:
        delta_acc = None
        delta_auc = None
        if direct_res is not None:
            log.info(
                "Direct-only run: Acc=%.4f AUC=%.4f",
                direct_res["accuracy"],
                direct_res["auc"],
            )
        if foundation_res is not None:
            log.info(
                "Foundation-only run: Acc=%.4f AUC=%.4f",
                foundation_res["accuracy"],
                foundation_res["auc"],
            )

    summary = {
        "run": args.run,
        "direct": (
            {
                "accuracy": float(direct_res["accuracy"]),
                "auc": float(direct_res["auc"]),
            }
            if direct_res is not None
            else None
        ),
        "foundation": (
            {
                "accuracy": float(foundation_res["accuracy"]),
                "auc": float(foundation_res["auc"]),
            }
            if foundation_res is not None
            else None
        ),
        "delta_accuracy": float(delta_acc) if delta_acc is not None else None,
        "delta_auc": float(delta_auc) if delta_auc is not None else None,
    }
    with (output_dir / "results.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    log.info("All outputs in %s", output_dir)


if __name__ == "__main__":
    main()