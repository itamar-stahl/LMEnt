# LMEnt 1B twin pair — a single-entity pretraining ablation

Two 1B models trained on the same Wikipedia corpus for the same one epoch,
identical in every respect **except** that one held every chunk mentioning
**Pornography (Wikidata `Q291`)** out of the training loss.

    lment-1b-control/   nothing held out       <- the baseline
    lment-1b-noporn/    Q291 held out          <- the ablated twin

They are only meaningful **as a pair**. The whole design is that they differ by
exactly the masked gradient contributions and by nothing else -- same data, same
order, same step count, same seed, same hardware. Comparing either one against
any other model, including the authors' released `dhgottesman/LMEnt-1B-1E`,
forfeits that guarantee.

Each directory has its own README with the details. Full documentation:

    git@github.com:itamar-stahl/LMEnt.git   branch itamars/main
    Untaught/STATUS.md      <- start here
    Untaught/RESULTS.md     <- what was trained, and proof the ablation fired
    ember_eval/EVALUATION.md <- what the evaluation found, and what it did not

Contact: Gal Barak <galll.barak@gmail.com>.
