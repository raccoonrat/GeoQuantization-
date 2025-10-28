



Recommendations for USENIXSecurity Submission
=============================================

1. Enhancing Experimental Reproducibility and Visualization

-----------------------------------------------------------

·         **Reproducibility:** Provide detailed dataset and experiment descriptions. Include code,model checkpoints, and configuration (e.g. random seeds, frameworks) in apublicly accessible repository, and list these artifacts in an “Open Science”appendix as required[[1]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=In%202025%2C%20USENIX%20Security%20introduced,code%20associated%20with%20research%20papers)[[2]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=In%202025%2C%20USENIX%20Security%20introduced,In%202026%2C%20authors). Report the hardware/software environment, hyperparameters, and anypreprocessing steps to enable exact replication.

·         **Quantization-errorvisualization:** Add figures that illustrate thequantization error for key layers or parameters. For example, plot thedistribution of weight/activation differences before and after quantization(e.g., histograms or scatter plots). This highlights how “functional outliers”differ from inliers under quantization. Show error vs. true value scatter orboxplots for different quantizers.

·         **HOP interpretability charts:** For the proposed HOP mechanism, include an intuitive visualization.For instance, use a heatmap or bar chart to show the sensitivity of outputlogits or features to perturbations in each quantized unit (higher orderparameter). This could identify which bits or weights most affect the model’spredictions.

·         **Privacy-accuracy trade-offplots:** Plot the model accuracy and privacy metricas functions of quantization bit-width. For example, on the x-axis usequantization bit-width (or quantization level) and on the y-axis show modelaccuracy (left axis) and a privacy measure (right axis), illustrating howincreasing precision improves utility but may reduce privacy.

·         **Membership-inference ROCcurves:** Include ROC curves for membership inferenceattacks against the quantized and DQC-protected models. The ROC (TPR vs FPR)shows attack success under different thresholds. For clarity, plot both linearand log-scaled versions. For example, see the ROC in the figure below; includesimilar curves comparing (a) full-precision vs. quantized vs. DQC models, and(b) different bits. This concretely demonstrates privacy gains.

_Figure: Example ROC curves for a baseline membership-inferenceattack (LOSS-based) on CIFAR-10_[_[3]_](https://arxiv.org/pdf/2112.03570#:~:text=Fig,rates%2C%20including%20high%20error%20rates)_. This illustrates how an attack’s true-positive rate (TPR) changeswith false-positive rate (FPR). In our submission, include comparable ROC plots(linear and log scales) to measure privacy leakage of quantized models._

·         **Quantization-error vs.outlier plots:** If the “functional outlier”definition involves a numerical threshold, show a chart of that function orthreshold. For example, plot the distribution of the chosen outlier score forall samples, marking the decision boundary. This helps readers see why certainpoints are deemed “outliers.”

·         **MIA vs. DP comparison:** When evaluating DQC, add plots comparing membership inferencesuccess for different DQC noise levels or DP noise parameters. For example,plot attacker accuracy (or AUC) versus the magnitude of injected DP noise, witherror bars. This quantifies the DP mechanism’s robustness.

·         **Statistical error bars:** Wherever performance metrics are reported (accuracy, privacyadvantage, HOP scores), include confidence intervals or standard deviationsacross multiple runs. For instance, error bars on accuracy/privacy plotshighlight stability.

·         **Appendix for data:** In a supplementary appendix, provide raw data tables or summarystatistics (e.g. dataset sizes, class distributions) used in experiments.Include pseudocode for data preprocessing if complex. Also document anyrandomization (seed setting) and how splits (train/test) are generated.

2. Completing Paper Structure and Content

-----------------------------------------

·         **Threat model (Introduction):** Explicitly state the adversary’s capabilities and goals. Forexample: “We assume an adversary has white-box access to the quantized model(or query access to its outputs) and seeks to infer sensitive training data viamembership inference or functional reconstruction. The adversary may observequantized weights/activations but cannot manipulate training.” This clarifiesassumptions (DP-style noise vs. malicious trainer) and scope. Relate this tostandard MI games[[4]](https://arxiv.org/pdf/2112.03570#:~:text=A,b%20%3D%200%2C%20samples%20a).

·         **Connections to latticetheory:** In Related Work, discuss how lattice orgeometric methods relate. If the outlier separability theory uses a lattice orgeometry interpretation, cite foundational work on lattice quantization orformal concept lattices. For example, one can mention optimal latticequantizers and their error bounds[[5]](https://arxiv.org/abs/2202.09605#:~:text=,optimal%20in%20the%20sense%20of) to contextualize approximation errors. Even if not cryptography,highlight any use of lattice structures or formal concept analysis in anomalydetection or quantization.

·         **Privacy defenses context:** Survey privacy-preserving quantization and noise mechanisms. Citerecent works that use quantization for DP: e.g. Youn _et al._ showrandomized quantization can achieve Rènyi-DP guarantees[[6]](https://arxiv.org/abs/2306.11913#:~:text=quantization%20to%20reduce%20communication%20complexity,To%20the%20best%20of%20our), and Yan _et al._ demonstrate that quantization noise providesinherent privacy benefits[[7]](https://arxiv.org/pdf/2304.13545#:~:text=We%20provided%20new%20insights%20into,REFERENCES). This situates “DQC privacy gain” within existing DP/quantizationliterature. Also mention classical DP references (e.g. Dwork and Roth) forcompleteness.

·         **Interpretability mechanisms:** Relate HOP to existing interpretability methods. Cite papers onfeature sensitivity, saliency maps, or activation clustering. If HOP is novel,explain differences from prior interpretable quantization or feature‐importancetechniques. For example, if it is akin to Layer-wise Relevance Propagation orsensitivity analysis, reference those to show novelty.

·         **Ethical considerations(Appendix):** Prepare a one-page “EthicalConsiderations” appendix as required[[8]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=%2A%20Papers%20must%20include%20,Overall%20length). Discuss any ethical implications of your work (e.g. privacy risks,dataset biases, potential misuse). Even if the work enhances privacy, note howmodel compression might affect fairness or security. Conclude by affirmingadherence to ethical guidelines.

·         **Open Science (Appendix):** Prepare the mandated “Open Science” appendix[[1]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=In%202025%2C%20USENIX%20Security%20introduced,code%20associated%20with%20research%20papers). List all artifacts: links to code repository, data sources, modelcheckpoints, and instructions. Confirm these will be available at submission.If any assets cannot be shared (e.g. proprietary data), explain why. Thisappendix ensures compliance with USENIX reproducibility policy.

·         **Algorithm pseudocode(Appendix):** Include clear pseudocode (in anappendix) for key methods: e.g., the HOP interpretation algorithm and the DQCquantization procedure. Label them as Algorithm A/B. This helps readers andreviewers verify correctness.

·         **Sensitivity analysis figures(Appendix):** Provide extra charts in the appendix,such as sensitivity vs. quantization bits for each layer or neuron. Forexample, a heatmap showing how output variance changes with small inputperturbations under different quantization schemes. This complements the maintext.

·         **Detailed metrics and stats(Appendix):** In an appendix, include any raw tablesor formula details omitted from the main text. For example, if you compute aprivacy metric (like MI advantage), show the exact statistical method orhypothesis test. Provide means and variances of multiple runs.

·         **Additional experiments(Appendix):** If space allows, include extra cases(e.g. other architectures or datasets) in the supplementary. Even if reviewersaren’t required to read them, appendices can answer detailed questions. Forinstance, a small section on how sensitive the results are to hyperparametersor on the computational overhead of DQC.

3. Language and Stylistic Polishing

-----------------------------------

·         **Professional tone:** Revise the English text for fluency. Use precise, formal languagetypical of security/ML papers. For example, replace colloquial phrases withtechnical terms (e.g. “we show” → “we demonstrate,” “bits” should be specifiedas “bit-widths” or “quantization bits”). Ensure each sentence clearly statesone idea.

·         **Logical flow:** Ensure smooth transitions and clear structure. For instance,explicitly connect the threat model to your goals: “Given this threat model,our goal is to ….” In Related Work, group citations thematically (e.g. “Priorwork on quantization-based privacy [36†L15-L24][38†L994-L1002]… Work on latticemethods [40†L50-L59]…”). Use linking phrases (“However,” “In contrast,”“Building on,” etc.) to clarify how your contributions differ.

·         **Terminology consistency:** Use consistent terms throughout. e.g., always say “quantizationbits” or “bit-width”, not both. Use standard terms for attacks (e.g. _membershipinference attack (MIA)_ with acronym defined on first use). Ensure “DP”,“Differential Privacy”, “DQC” are defined when first introduced.

·         **Conciseness and clarity:** Remove redundancies. For example, avoid repeating definitionsalready given; instead refer to the relevant section or figure. Ensure variablenames and notation are consistent (e.g. if you use $f_\theta$ for a model, keepit throughout). Spell out acronyms on first use.

·         **USENIX style:** Follow the two-column, 10-point Times font format. Use USENIX’ssectioning: Introduction, Background/Model, Methodology, Evaluation,Discussion, Conclusion. Number figures and tables consecutively, and useprofessional captioning (e.g. “Figure 1: …”). Number equations ifreferenced. For citations, use bracketed numbers (e.g. [15]) per USENIX style.

·         **Grammar/Spelling:** Carefully proofread for grammar and spelling. USENIX reviewersexpect near-perfect English. Read aloud or use grammar tools to catch issues(e.g. plural agreement, article usage). In Chinese-to-English translation,ensure idioms are correctly rendered (e.g. “functionally sensitive outliers”rather than literal translations).

4. Anticipating Reviewer Questions and Responses

------------------------------------------------

·         **Robustness of DP noise:** _Possible question:_ _“How sensitive is DQC privacy gain tothe choice of noise distribution or parameters?”_ **Response:** Emphasizethat we compare DQC against standard DP baselines. Cite [35] and [38] to arguethat quantization inherently provides DP-like protection (Rènyi-DP) and thatour added noise can be tuned with provable bounds. Mention any empirical testsshowing privacy metrics (e.g. membership inference AUC) remain stable undersmall changes in noise. If possible, reference theoretical guarantees from DPliterature (e.g. the Laplace or binomial mechanisms) to show robustness.

·         **Lattice approximation errorbounds:** _Possible question:_ _“What is theerror bound when approximating with your lattice-based (or geometric) method?”_ **Response:** Explain that our “separability geometry” analysis relies onknown quantization error results. For example, cite Agrell & Allen’s boundson lattice quantizers[[5]](https://arxiv.org/abs/2202.09605#:~:text=,optimal%20in%20the%20sense%20of) to argue that our error is at most some fraction of thequantization cell size. If we derived a bound, outline it succinctly (perhapsin an appendix). Emphasize that worst-case errors are bounded by design andthat in practice we observe small approximation error (supported by ourcharts).

·         **Adversarialexamples/robustness:** _Possible question:_ _“Howdoes the method generalize to adversarially crafted inputs?”_ **Response:** Clarify that our focus is on _statistical_ outliers and privacy leakage,not on evading adversarial attacks. However, one can argue that DQC’s addednoise might incidentally increase robustness against small adversarialperturbations (by reducing precision). Cite any relevant work (e.g. on quantizednetworks and robustness). If no direct evaluation was done, acknowledge this asfuture work. Also note that HOP interpretability could flag inputs withabnormal activation patterns (similar to some defenses). If applicable, mentionany preliminary tests (even if brief) that show how accuracy drops underadversarial noise for quantized vs. unquantized models.

·         **HOP novelty and baseline:** _Possible question:_ _“How is your HOP method different fromstandard sensitivity analysis or feature importance methods?”_ **Response:** Emphasize any novel aspect of HOP (e.g. focusing on higher-order combinationsof bits or layers). Compare briefly to known methods (LRP, SHAP, etc.) andargue why those don’t directly apply to quantized networks. If HOP uses a newmetric (like a special distance or score), explain why prior metrics wouldn’tcapture the same insight. Possibly include a small example (in an appendix)showing how HOP identifies features that standard methods miss.

·         **Functional outlierdefinition validity:** _Possible question:_ _“Whyuse this particular definition of a functional outlier? How does it compare toother definitions?”_ **Response:** Argue that our definition is motivatedby its relevance to model behavior under quantization. We could compare itconceptually to statistical outliers or adversarial examples. For example, saythat unlike raw input outliers, “functional outliers” are those that produceunusually large changes in output under quantization. If any baseline methodsexist (e.g. computing influence functions), mention that as a contrast. Ifpossible, refer to an example or cite literature on outliers in model space(though if none, emphasize the novelty).

·         **Scalability and overhead:** _Possible question:_ _“What is the computational cost of DQCand HOP? Do they scale to large models?”_ **Response:** Provide anymeasured runtime or complexity analysis. For example, state the bit-width ofDQC (4-8 bits) has minimal overhead in hardware, and HOP’s sensitivitycomputation is linear in network size. If GPU/TPU timings were measured, reportthem (or at least say “on ResNet-50 it adds ~X% overhead”). Compare tobaselines (e.g. “comparable to standard PTQ”). Emphasize that all experimentswere done with standard libraries, implying practicality.

·         **Comparison to related work:** _Possible question:_ _“How do your results compare to [35],[36], or other recent quantization-privacy schemes?”_ **Response:** Makesure to include in related work or evaluation any relevant methods. Forexample, if [36] (BQ-SGD) or [35] (RQM) provide DP for quantization, mentionthem in discussion: we evaluate similar metrics (e.g. membership inferencesuccess) to show our DQC performs equally or better in accuracy-privacytradeoff. If theirs are in federated learning, explain that our setting isdifferent, but qualitatively the trade-offs align. If we can, add a comparisontable or at least a qualitative statement.

5. USENIX Template and Formatting Guidelines

--------------------------------------------

·         **Official Template:** Use the USENIX Security LaTeX template without modifications[[9]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=,profile%20page%20on%20HotCRP%20by). Do _not_ alter margins or spacing (no \savetrees or negative vspaces). Thepaper should be in two-column format, 10pt Times font, on US letter paper[[10]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=10,Authors%20must%20use%20the). References must be in USENIX style (numeric [1]–[N] in brackets).Authors and acknowledgments should be omitted for blind review.

·         **Page limits:** The main text (Sections 1–6) must be ≤13 pages[[11]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=,each%20one%20page%20of%20text). Mandatory appendices (“Ethical Considerations” and “Open Science”)each get up to 1 page[[11]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=,each%20one%20page%20of%20text). Any extra appendices beyond these can be longer, but reviewersaren’t required to read them[[12]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=At%20submission%20time%2C%20there%20is,contained%20without%20appendices)[[11]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=,each%20one%20page%20of%20text).

·         **Section structure:** Follow a logical section order. A typical outline is: **Abstract**, **Introduction** (with motivation and threat model), **Related Work**, **Background/Definitions** (if needed), **Methodology** (outlier geometry, HOP, DQC definitions), **Evaluation/Experiments**, **Discussion**, **Conclusion**. After the main text, include the **EthicalConsiderations** and **Open Science** appendices per the CFP. Then listreferences. You may optionally include additional appendices (e.g. proofs,extra figures).

·         **Figures and tables:** Place figures/tables in the paper where first cited. Use the figure* environment for wide figures ifneeded, and ensure all graphics are legible in grayscale[[13]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=figures%2C%20are%20intelligible%20when%20printed,in%20grayscale). Provide descriptive captions (e.g. “Figure 3: Accuracy vs.bit-width on CIFAR-10.”). Number them consecutively and refer to them in textas “Fig. 1,” etc. Use high-resolution images or vector graphics to avoidpixelation. Subfigures (with (a), (b) labels) are acceptable for multi-panelplots.

·         **Algorithms and pseudocode:** Use the algorithm or algorithmic environment (or algorithm2e) for pseudocode. Numberalgorithms and refer to them in text. Make sure any math symbols used aredefined.

·         **Equations and references:** Number only those equations you reference in the text. Alignequations for readability. For citations, use \cite{} which will appear as [5] style.In text, refer to citations by number, e.g., “as shown in [12].” All referencesmust be complete and formatted per USENIX style. For example: _B. Author,“Title,” in Proceedings of USENIX Security ’20, 2020._

·         **Miscellaneous:** Ensure consistent notation (bold for vectors, _italics_ forvariables, etc.), and consistent heading capitalization (title-case for sectionheadings). Use Latin abbreviations correctly (e.g. “et al.”, not “& al.”).Check that acronyms are explained and used consistently. Finally, compile withPDFLaTeX and verify the PDF is compliant (no page overflows, all fontsembedded, and figures readable in black-and-white).

**Proposed Appendices:**  

- _Ethical Considerations_: A formal, stakeholder-based analysis of ethics(per CFP guidelines), even if the work aims to enhance privacy.  
- _Open Science_: A list of all shared artifacts (code, data, models) andhow to access them[[1]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=In%202025%2C%20USENIX%20Security%20introduced,code%20associated%20with%20research%20papers).  
- _Pseudo-code Algorithms_: Detailed algorithms (HOP feature importance,DQC quantizer, sensitivity scoring, etc.).  
- _Data Statistics and Preprocessing_: Tables of dataset characteristics(size, classes), and any preprocessing or calibration procedures.  
- _Additional Figures_: Extra charts such as layer-wise sensitivity plots,extended ROC curves for different thresholds, or bit-precision vs. metric plotsnot in the main text.  
- _Sensitivity Analysis_: Illustrative examples showing how small input orweight perturbations affect outputs under quantization (could be a figure ortable).  
- _Proof Sketches or Derivations_: Any mathematical proofs or derivationsomitted for brevity (e.g. bound on quantization error).  
- _User Studies or Extended Results_: If applicable, any other evaluations(e.g. on more datasets) can be placed here.

Each of the above should be clearly labeled as appendices A, B, C,etc., using the USENIX appendix formatting. This completes a thorough plan tomeet USENIX Security standards.

**Sources:** USENIX CFP and submissioninstructions[[14]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=Submitted%20papers%20should%20describe%20original%2C,to%20appear%20in%20the%20program)[[9]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=,profile%20page%20on%20HotCRP%20by)[[11]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=,each%20one%20page%20of%20text)[[10]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=10,Authors%20must%20use%20the); recent quantization/privacy literature[[6]](https://arxiv.org/abs/2306.11913#:~:text=quantization%20to%20reduce%20communication%20complexity,To%20the%20best%20of%20our)[[7]](https://arxiv.org/pdf/2304.13545#:~:text=We%20provided%20new%20insights%20into,REFERENCES); privacy evaluation best practices[[3]](https://arxiv.org/pdf/2112.03570#:~:text=Fig,rates%2C%20including%20high%20error%20rates).

* * *

[[1]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=In%202025%2C%20USENIX%20Security%20introduced,code%20associated%20with%20research%20papers) [[2]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=In%202025%2C%20USENIX%20Security%20introduced,In%202026%2C%20authors) [[8]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=%2A%20Papers%20must%20include%20,Overall%20length) [[9]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=,profile%20page%20on%20HotCRP%20by) [[10]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=10,Authors%20must%20use%20the) [[11]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=,each%20one%20page%20of%20text) [[12]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=At%20submission%20time%2C%20there%20is,contained%20without%20appendices) [[13]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=figures%2C%20are%20intelligible%20when%20printed,in%20grayscale) [[14]](https://www.usenix.org/conference/usenixsecurity26/call-for-papers#:~:text=Submitted%20papers%20should%20describe%20original%2C,to%20appear%20in%20the%20program) USENIX Security '26 Call for Papers | USENIX

[USENIX Security &#039;26 Call for Papers | USENIX](https://www.usenix.org/conference/usenixsecurity26/call-for-papers)

[[3]](https://arxiv.org/pdf/2112.03570#:~:text=Fig,rates%2C%20including%20high%20error%20rates) [[4]](https://arxiv.org/pdf/2112.03570#:~:text=A,b%20%3D%200%2C%20samples%20a) arxiv.org

https://arxiv.org/pdf/2112.03570

[[5]](https://arxiv.org/abs/2202.09605#:~:text=,optimal%20in%20the%20sense%20of) [2202.09605] On the best lattice quantizers

[[2202.09605] On the best lattice quantizers](https://arxiv.org/abs/2202.09605)

[[6]](https://arxiv.org/abs/2306.11913#:~:text=quantization%20to%20reduce%20communication%20complexity,To%20the%20best%20of%20our) [2306.11913] Randomized Quantization is All You Need forDifferential Privacy in Federated Learning

[[2306.11913] Randomized Quantization is All You Need for Differential Privacy in Federated Learning](https://arxiv.org/abs/2306.11913)

[[7]](https://arxiv.org/pdf/2304.13545#:~:text=We%20provided%20new%20insights%20into,REFERENCES) arxiv.org

https://arxiv.org/pdf/2304.13545
