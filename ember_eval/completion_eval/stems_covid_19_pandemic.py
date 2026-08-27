#!/usr/bin/env python
"""Declarative-stem rewrites of EMBER's COVID-19 pandemic questions.

Same rules as stems_pornography.py, which is the reference copy of them:

  1. Declarative prose, no question mark, no "Question:".
  2. All four options continue the stem grammatically.
  3. Written from the question, never from the answer key.
  4. Stop before a determiner unless all four options take it.
  5. Options, answer key, file order and shuffle seed unchanged.

The concept split is mostly numbers, dates and drug names, so a plain "...was"
frame carries almost all of it. The similar-domain split is general medicine and
is where the two interesting problems live.

**Polar questions.** Eight similar-domain items are yes/no or either/or, with
options like `No | Yes | Only for children | Only for the elderly`. No noun
phrase frame takes all four, so these use a reported-answer frame:

    On the question of whether there is a vaccine available for the common
    cold, the answer is ___

which takes No, Yes, Only for children and Only for the elderly equally. Note
what these items can and cannot measure. A one-token option carries almost no
content of its own, so what is being scored is close to the model's bare
Yes/No prior under that context, and the null-context subtraction is doing more
work than usual. They are still valid items, in that preferring "No" after the
cold-vaccine context does require knowing there is no cold vaccine, but expect
them to be the noisiest eight in the concept. They are EMBER's items and are
left in place. The eight are SimdomQA_train #2, #3, #16, #45 and SimdomQA_test
#24, #31, #41, #42.

**Rule 4.** Vowel-initial options are unusually common in the medical
vocabulary here, and each one kills an "...is a" ending: Epidemic against
Pandemic/Endemic/Sporadic, Antigen against Antibody/Pathogen/Platelet, Amoeba
in the pathogen set, Antibiotic in the vaccine set, Inhaler in the lozenge set.
Goggles kills it a different way, by being plural. Those seven stems stop at
"is" or "is called". Where all four take an article the stem keeps it, as in
Neutrophil / Lymphocyte / Monocyte / Eosinophil, all of which take "the".

No deviations from the original wording. Every stem is built from content words
already in its question.

Two morphological near-misses worth flagging, since they look like leaks and are
not. QA_train #15 must say "variants" and never "variant", because the gold
option is *Variant-adapted vaccines* and no distractor contains "variant".
SimdomQA_test #43 must say "yellowing" and never "yellow", for the same reason
against *Yellow Fever*. `build_completions.py --check` catches both if a later
edit slips.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# The eight polar items. NOT SURE THESE ARE USABLE - see the note below.
#
# Each entry below is also marked inline with `# POLAR`. Read this before
# excluding them, because deleting the entries is the wrong way to do it.
#
# Why they are doubtful: the options are `No | Yes | Only for children | ...`.
# A one-token option carries almost no content of its own, so the conditional
# log-probability is driven mostly by the model's generic preference for "No"
# in that slot rather than by anything it knows about colds or flu. The pmi
# metric helps more here than anywhere else, since subtracting
# log P(a | null) cancels a global Yes/No prior, so the items are not hollow.
# But the signal is thin, and five of the eight golds are "No", so a model that
# simply prefers "No" scores 5/8 without knowing anything.
#
# How to exclude them: filter at analysis time, do not delete these entries.
# Deleting breaks two things. build_completions.py requires exactly 50 stems
# per split and will refuse to build with 46. More importantly, dropping
# questions renumbers the split and breaks the guarantee the whole package
# rests on, that item N here is item N in every multiple-choice run already
# recorded. Instead:
#
#     from stems_covid_19_pandemic import POLAR_ITEMS
#     rows = [r for r in results["results"] if r["question"] not in POLAR_ITEMS]
#
# Every row in an evaluate_completion.py results file carries its `question`
# string, so that filter needs no changes anywhere else.
#
# Suggested first move: evaluate all 200, then compare the twins on these eight
# against the other 42 in the same split. If they behave like the rest, keep
# them. If they are noise, you have a documented reason to drop them.
# --------------------------------------------------------------------------- #
POLAR_ITEMS: frozenset[str] = frozenset({
    "Is there a vaccine available for the common cold?",
    "Does the flu usually cause a high fever?",
    "Can the flu be transmitted before a person shows symptoms?",
    "Are colds more common in children or adults?",
    "Does sweating heavily cure a cold?",
    "Is the 'stomach flu' the same virus as influenza?",
    "Does simply being in cold weather cause a cold?",
    "Can you cure the flu with antibiotics?",
})

STEMS: dict[str, list[tuple[str, str]]] = {}

# --------------------------------------------------------------------------- #
# Concept QA - validation
# --------------------------------------------------------------------------- #
STEMS["QA_train"] = [
    ("What subgenus does the SARS-CoV-2 virus",
     "The subgenus that the SARS-CoV-2 virus genetically clusters with is"),
    ("How long do mild COVID-19 cases",
     "The time mild COVID-19 cases typically take to recover is"),
    ("What country had the highest number of confirmed cases",
     "By 26 March 2020, the country with the highest number of confirmed cases "
     "in the world was"),
    ("Genetic variants at what chromosomal region",
     "Genetic variants associated with Neanderthal heritage that increase "
     "vulnerability to severe COVID-19 are located at"),
    ("What was the metaregression estimate of the infection fatality rate for an 85",
     "The metaregression estimate of the infection fatality rate for an "
     "85-year-old adult was"),
    ("What surface glycoprotein does the virus use",
     "The surface glycoprotein the virus uses to connect to the ACE2 receptor is the"),
    ("What is the critical diameter size",
     "The critical diameter size that distinguishes large respiratory droplets "
     "from small aerosol particles is"),
    ("What is the marketed brand name",
     "The marketed brand name of the nirmatrelvir/ritonavir medication used for "
     "mild to moderate symptoms is"),
    ("What trial did the WHO initiate",
     "In March 2020, to assess the treatment effects of promising drugs, the "
     "WHO initiated the"),
    ("On what date did the WHO declare COVID-19 a Public Health Emergency",
     "The WHO declared COVID-19 a Public Health Emergency of International "
     "Concern on"),
    ("What percentage of acute cardiac injury",
     "The percentage of acute cardiac injury found in infected people admitted "
     "to the hospital in Wuhan was"),
    ("Men with what untreated condition",
     "Men were 2.4 times more likely to be hospitalized if they contracted "
     "COVID-19 when they had the untreated condition"),
    ("What was the metaregression estimate of the infection fatality rate for a 10",
     "The metaregression estimate of the infection fatality rate for a "
     "10-year-old child was"),
    ("What is the median delay or incubation period",
     "The median delay or incubation period for COVID-19 is"),
    ("What updated vaccine doses are offered",
     "The updated vaccine doses offered to combat new SARS-CoV-2 variants are"),
    ("What country overtook China",
     "On 19 March 2020, the country that overtook China as the country with the "
     "most COVID-19 deaths was"),
    ("What percentage of the SARS-CoV-2 genome",
     "The percentage of the SARS-CoV-2 genome identical to the bat coronavirus "
     "sample BatCov RaTG13 is"),
    ("What animal, a major pelt producer",
     "The animal, a major pelt producer in Denmark, that was slaughtered over "
     "fears of viral mutations was the"),
    ("In patients requiring hospital admission",
     "In patients requiring hospital admission, the percentage of CT scans "
     "showing lung abnormalities after 28 days is"),
    ("How long do severe or critical COVID-19 cases",
     "The time severe or critical COVID-19 cases typically take to recover is"),
    ("What is the condition characterized by persistent symptoms",
     "The condition characterized by persistent symptoms for months or years "
     "after infection is called"),
    ("What percentage of symptomatic cases report loss of taste",
     "The percentage of symptomatic cases reporting loss of taste combined with "
     "loss of smell is"),
    ("In what month and year was the first known case identified",
     "The first known case was identified in"),
    ("When did the World Health Organization declare COVID-19 a global health emergency",
     "The World Health Organization declared COVID-19 a global health emergency in"),
    ("What earliest date of symptom onset",
     "The earliest date of symptom onset reported by official publications from "
     "the WHO was"),
    ("What interleukin-6 receptor antagonist",
     "The interleukin-6 receptor antagonist that underwent a Phase III clinical "
     "trial to assess its effectiveness on COVID-19 was"),
    ("When did the World Health Organization declare the end",
     "The World Health Organization declared the end of the global health "
     "emergency in"),
    ("What percentage of symptomatic patients develop mild",
     "The percentage of symptomatic patients who develop mild to moderate "
     "symptoms is"),
    ("What minimum fraction of infected people",
     "The minimum fraction of infected people who do not develop noticeable "
     "symptoms is"),
    ("How many days can people remain contagious",
     "The length of time people can remain contagious with the virus is"),
    ("What is the minimum alcohol percentage",
     "The minimum alcohol percentage recommended by the CDC for hand sanitizer is"),
    ("On what date was the first COVID-19 vaccine granted regulatory approval",
     "The first COVID-19 vaccine was granted regulatory approval by the UK "
     "medicines regulator MHRA on"),
    ("What mortality rate was found in a study of hospitalized kidney transplant",
     "The mortality rate found in a study of hospitalized kidney transplant "
     "recipients with COVID-19 was"),
    ("What common household item",
     "The common household item that destroys the virus outside the human body "
     "by bursting its protective bubble is"),
    ("What is the primary target for neutralizing antibodies",
     "On the SARS-CoV-2 virus, the primary target for neutralizing antibodies is the"),
    ("What was the median time between the onset of symptoms and death",
     "The median time between the onset of symptoms and death reported by the "
     "Italian Istituto Superiore di Sanità was"),
    ("What percentage of hospitalized children",
     "According to a European multinational study from June 2020, the "
     "percentage of hospitalized children who needed intensive care was"),
    ("On what date was human-to-human transmission confirmed",
     "Human-to-human transmission was confirmed by the WHO and Chinese "
     "authorities on"),
    ("What percentage of healthy adults",
     "The percentage of healthy adults not exposed to SARS-CoV-2 who have CD4+ "
     "T cells that recognize the S protein is"),
    ("What liquid portion of the blood",
     "The liquid portion of the blood from recovered patients used in passive "
     "antibody therapy is"),
    ("Up to what percentage of hospitalized people in China and New York",
     "Early in the pandemic, the percentage of hospitalized people in China and "
     "New York who experienced kidney injury was"),
    ("What percentage of symptomatic patients develop severe",
     "The percentage of symptomatic patients who develop severe symptoms "
     "requiring hospitalization is"),
    ("What humectant is added",
     "The humectant added to WHO hand sanitizer formulations is"),
    ("What variant became dominant",
     "The variant that became dominant in the U.S. in December 2021 was the"),
    ("What laboratory method detects",
     "The laboratory method that detects the presence of viral RNA fragments is"),
    ("What is the more accurate test",
     "The more accurate test, the one typically analyzed in a laboratory, is the"),
    ("What UK trial showed in June 2020",
     "The UK trial that showed in June 2020 that dexamethasone reduced "
     "mortality for critically ill patients was the"),
    ("What percentage of COVID-19 cases are severe enough",
     "The percentage of COVID-19 cases severe enough to cause hospitalization is"),
    ("What type of infection are aspergillosis",
     "Aspergillosis, candidiasis, and mucormycosis are examples of"),
    ("How many seconds does the WHO recommend",
     "The WHO recommends washing hands with soap and water for"),
]

# --------------------------------------------------------------------------- #
# Concept QA - test
# --------------------------------------------------------------------------- #
STEMS["QA_test"] = [
    ("In what month and year was the first case of reinfection",
     "The first case of reinfection was documented in"),
    ("What is the medical term for the fluid patches",
     "The medical term for the fluid patches observed in CT scans of COVID-19 "
     "infected lungs is"),
    ("Abnormal levels of what mineral",
     "Abnormal levels during hospitalization associated with poor prognoses are "
     "found in the mineral"),
    ("What percentage of 106,000 discharged individuals",
     "The percentage of 106,000 discharged individuals who had to return for "
     "hospital treatment within two months was"),
    ("What enzyme receptor does the virus use",
     "The enzyme receptor the virus uses to access host cells is"),
    ("What drug was approved by the UK in November 2021",
     "The drug approved by the UK in November 2021 as a treatment for "
     "vulnerable patients recently diagnosed with COVID-19 was"),
    ("What death rate for women",
     "The death rate for women reported by the Chinese Center for Disease "
     "Control and Prevention was"),
    ("What virus causes Coronavirus disease 2019",
     "The virus that causes Coronavirus disease 2019 is"),
    ("What US initiative allows people",
     "The US initiative that allows people to take a COVID test at a pharmacy "
     "and immediately receive free Paxlovid is"),
    ("What is COVID-19 shorthand for",
     "COVID-19 is shorthand for"),
    ("How many vaccine doses had been preordered",
     "By December 2020, the number of vaccine doses preordered globally was"),
    ("How many COVID-19 vaccine doses had been administered",
     "As of August 2024, the number of COVID-19 vaccine doses administered "
     "worldwide was"),
    ("What was the global death-to-case ratio",
     "Based on Johns Hopkins University statistics, the global death-to-case "
     "ratio as of 10 March 2023 was"),
    ("In what city and country was the first known COVID-19 case",
     "The first known COVID-19 case was identified in"),
    ("What type of COVID test is also called",
     "The type of COVID test also called a rapid lateral flow test is the"),
    ("What animal was associated with the Cluster 5 variant",
     "The animal associated with the Cluster 5 variant in Denmark was the"),
    ("What specific animal species",
     "The specific animal species that has a virus containing all structural "
     "features of the novel SARS-CoV-2 virus is"),
    ("On what date did the Public Health Emergency of International Concern",
     "The Public Health Emergency of International Concern for COVID-19 ended on"),
    ("In March 2020, what percentage of hospitalized people",
     "In March 2020, the percentage of hospitalized people in the United States "
     "who had preexisting conditions was"),
    ("What blood group was initially suggested",
     "The blood group initially suggested by scientists in Wuhan to be more "
     "likely to experience severe symptoms was"),
    ("What antimalarial drug showed it was ineffective",
     "The antimalarial drug that showed it was ineffective at best and was "
     "banned as a treatment in France, Italy, and Belgium by May 2020 was"),
    ("What are the four structural proteins",
     "The four structural proteins of SARS-CoV-2 are"),
    ("What percentage of American Indian and Alaska Native",
     "According to a US health policy non-profit, the percentage of American "
     "Indian and Alaska Native non-elderly adults at risk of serious illness is"),
    ("According to the Italian Istituto Superiore di Sanità, what was the most common",
     "According to the Italian Istituto Superiore di Sanità, the most common "
     "comorbidity among COVID-19 deaths was"),
    ("Who was the doctor in Hubei Provincial Hospital",
     "The doctor in Hubei Provincial Hospital who observed a pneumonia cluster "
     "of unknown cause on 26 December 2019 was"),
    ("What is the term for slowing the infection rate",
     "The term for slowing the infection rate to decrease the risk of health "
     "services being overwhelmed is"),
    ("What emergency ICD-10 disease code",
     "For deaths from lab-confirmed SARS-CoV-2 infection, the emergency ICD-10 "
     "disease code assigned by the WHO was"),
    ("What is the minimum weight requirement",
     "According to the European Medicines Agency endorsement, the minimum "
     "weight requirement for the use of dexamethasone is"),
    ("In a January 2020 Lancet study",
     "In a January 2020 Lancet study of the first 41 cases, the earliest date "
     "of symptom onset was"),
    ("What common symptom results from the infection of the support cells",
     "The common symptom that results from the infection of the support cells "
     "of the olfactory epithelium is"),
    ("What is the incidence rate of long-term effects",
     "The incidence rate of long-term effects for COVID-19 patients who need "
     "hospitalization is"),
    ("What are the two subunits",
     "The two subunits of the spike protein are"),
    ("What glucocorticoid medication",
     "The glucocorticoid medication strongly recommended in November 2020 for "
     "severe cases treated in hospital with low oxygen levels was"),
    ("What antiviral drug has been discouraged",
     "The antiviral drug discouraged by the WHO due to limited evidence of "
     "efficacy is"),
    ("What gene allele is a common risk factor",
     "The gene allele that is a common risk factor for severe COVID-19 in Asian "
     "populations is the"),
    ("What acute hyperinflammatory response",
     "The acute hyperinflammatory response related to worse prognosis and "
     "increased fatality in COVID-19 is"),
    ("In what month and year did the COVID-19 disease start spreading",
     "The COVID-19 disease started spreading worldwide in"),
    ("What percentage of symptomatic patients develop critical",
     "The percentage of symptomatic patients who develop critical symptoms like "
     "respiratory failure is"),
    ("What variant did the WHO declare",
     "On 19 December 2023, the variant the WHO declared a 'variant of interest' was"),
    ("What is the maximum number of days",
     "The maximum number of days it may take for symptoms to begin after "
     "exposure to the virus is"),
    ("What metric divides the total number of deaths",
     "The metric that divides the total number of deaths by the total number of "
     "infected individuals, including asymptomatic ones, is the"),
    ("Within how many days do most symptomatic people",
     "Most symptomatic people experience symptoms after exposure within"),
    ("What alphabet did the WHO recommend",
     "To label virus variants, the WHO recommended using the"),
    ("What metric reflects the number of deaths",
     "The metric that reflects the number of deaths divided by the number of "
     "diagnosed cases within a given time interval is the"),
    ("How long are individuals advised to self-quarantine",
     "From the time of last possible exposure, individuals are advised to "
     "self-quarantine for"),
    ("In August 2020, what bacterial lung infection",
     "In August 2020, the bacterial lung infection the CDC cautioned could "
     "increase the risk of severe illness or death was"),
    ("What date did the WHO issue the official names",
     "The WHO issued the official names COVID-19 and SARS-CoV-2 on"),
    ("According to Reuters",
     "According to Reuters, the worldwide total number of deaths exceeded five "
     "million by"),
    ("According to a global meta-analysis",
     "According to a global meta-analysis, the number per 100 infections who "
     "get COVID-19 compared to women is"),
    ("According to July 2022 Science articles",
     "According to July 2022 Science articles, the two spillover events at the "
     "Huanan wet market occurred in"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - validation
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_train"] = [
    ("What virus is usually transmitted through the bite",
     "The virus usually transmitted through the bite of an infected, rabid "
     "animal is"),
    # POLAR - not sure we can use this one; see POLAR_ITEMS at the top
    #         5 of 8 golds are "No"
    ("Is there a vaccine available for the common cold",
     "On the question of whether there is a vaccine available for the common "
     "cold, the answer is"),
    # POLAR - not sure we can use this one; see POLAR_ITEMS at the top
    ("Does the flu usually cause a high fever",
     "On the question of whether the flu usually causes a high fever, the answer is"),
    ("What shape are cocci bacteria",
     "The shape of cocci bacteria is"),
    ("What virus family causes the flu",
     "The virus family that causes the flu is"),
    ("What is the medical term for a runny nose",
     "The medical term for a runny nose is"),
    ("What is any foreign substance",
     "Any foreign substance that triggers an immune response is called"),
    ("What shape are bacilli bacteria",
     "The shape of bacilli bacteria is"),
    ("What are the beneficial bacteria",
     "The beneficial bacteria that live in the human digestive tract are called"),
    ("What viral infection of the liver",
     "The viral infection of the liver commonly spread through contaminated "
     "food or water is"),
    ("What is the deadliest strain of malaria",
     "The deadliest strain of malaria is"),
    ("What very common skin condition",
     "The very common skin condition caused by bacteria trapped in pores is"),
    ("What is the medical term for the swelling",
     "The medical term for the swelling, redness, and heat at an injury site is"),
    ("What is the body's largest physical barrier",
     "The body's largest physical barrier and first line of defense against "
     "pathogens is"),
    ("What disease causes a severe, hacking cough",
     "The disease that causes a severe, hacking cough followed by a "
     "high-pitched intake of breath is"),
    # POLAR - not sure we can use this one; see POLAR_ITEMS at the top
    ("Can the flu be transmitted before a person shows symptoms",
     "On the question of whether the flu can be transmitted before a person "
     "shows symptoms, the answer is"),
    ("What part of the body does a cold primarily affect",
     "The part of the body a cold primarily affects is the"),
    ("Who discovered the first antibiotic",
     "The first antibiotic, penicillin, was discovered by"),
    ("What organ filters blood",
     "The organ that filters blood and helps fight infections is the"),
    ("How long does a typical uncomplicated flu last",
     "A typical uncomplicated flu lasts"),
    ("What simple hygiene habit",
     "The simple hygiene habit considered the most effective way to stop the "
     "spread of germs is"),
    ("What classic warm liquid",
     "The classic warm liquid that is a common home remedy for a cold is"),
    ("What is the most frequent viral cause",
     "The most frequent viral cause of the common cold is"),
    ("What are single-celled organisms",
     "Single-celled organisms that can live almost anywhere, some of which "
     "cause disease, are"),
    ("What gland, located behind the sternum",
     "The gland located behind the sternum responsible for maturing T-cells is the"),
    ("What is the medical term for muscle aches",
     "The medical term for muscle aches, which are common during the flu, is"),
    ("What wild bird",
     "The wild bird that is a natural host for many influenza viruses is the"),
    ("What scientist developed the first successful polio vaccine",
     "The first successful polio vaccine was developed by the scientist"),
    ("What are the tiny hairs",
     "The tiny hairs in the respiratory tract that sweep away mucus and dirt are"),
    ("How many times a year does the average adult get a cold",
     "The average adult gets a cold"),
    ("What medical device is used to inject medicine",
     "The medical device used to inject medicine or vaccines into the body is a"),
    ("What vitamin is popularly taken",
     "The vitamin popularly taken to try and shorten a cold is"),
    ("What type of severe, life-threatening allergic reaction",
     "The type of severe, life-threatening allergic reaction that requires "
     "immediate medical attention is"),
    ("What is the process called when a white blood cell",
     "When a white blood cell engulfs and digests a bacterium, the process is called"),
    ("What common foodborne illness",
     "The common foodborne illness frequently associated with eating "
     "undercooked poultry is"),
    ("What type of organism causes ringworm",
     "The type of organism that causes ringworm is a"),
    ("How is the flu primarily spread",
     "The flu is primarily spread by"),
    ("What cells fight infection",
     "The cells that fight infection in the human body are"),
    ("What alcohol-based liquid",
     "The alcohol-based liquid used to kill germs on hands when soap and water "
     "are unavailable is"),
    ("What component of blood helps it clot",
     "The component of blood that helps it clot to prevent infection in an open "
     "wound is"),
    ("What bacterial infection commonly causes a very sore",
     "The bacterial infection that commonly causes a very sore, red throat and "
     "hurts to swallow is"),
    ("What disease famously paralyzed",
     "The disease that famously paralyzed US President Franklin D. Roosevelt was"),
    ("What do we call a sudden, localized outbreak",
     "A sudden, localized outbreak of a disease in a specific community is called"),
    ("What type of immune cells remember",
     "The type of immune cells that remember past infections to respond faster "
     "next time are"),
    # POLAR - not sure we can use this one; see POLAR_ITEMS at the top
    #         also a word-match item: "children" is in the question and the gold
    ("Are colds more common in children or adults",
     "On the question of whether colds are more common in children or adults, "
     "the answer is"),
    ("What is the common, informal name",
     "The common, informal name for the severe muscle spasms caused by tetanus is"),
    ("What sticky bodily secretion",
     "The sticky bodily secretion that traps dust and pathogens in the nasal "
     "passage is"),
    ("What famous nurse",
     "The famous nurse who dramatically improved hygiene practices during the "
     "Crimean War was"),
    ("What small, bean-shaped structures",
     "The small, bean-shaped structures that filter lymph fluid throughout the "
     "body are"),
    ("What action clears mucus",
     "The action that clears mucus from the lower airways is"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - test
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_test"] = [
    ("What pandemic killed roughly a third",
     "The pandemic that killed roughly a third of Europe's population in the "
     "14th century was"),
    ("What sudden, forceful action",
     "The sudden, forceful action that clears irritants from the nasal cavity is"),
    ("What common symptom is the body's way",
     "The common symptom that is the body's way of forcefully expelling "
     "irritants from the lungs is a"),
    ("What virus caused a major outbreak in West Africa",
     "The virus that caused a major outbreak in West Africa between 2014 and "
     "2016 was"),
    ("What type of immunity is passed",
     "The type of immunity passed from a mother to her baby is"),
    ("What type of medicine is specifically used",
     "The type of medicine specifically used to treat fungal infections is"),
    ("What bacterium causes the Bubonic plague",
     "The bacterium that causes the Bubonic plague is"),
    ("What is the scientific study of how diseases spread",
     "The scientific study of how diseases spread and can be controlled is"),
    ("What is the recommended preventative measure",
     "The recommended preventative measure taken annually for the flu is"),
    ("What insect transmits the Zika virus",
     "The insect that transmits the Zika virus is the"),
    ("What farm animal",
     "The farm animal that is a common mixing vessel for human, avian, and "
     "swine flu viruses is the"),
    ("What is the most abundant type of white blood cell",
     "The most abundant type of white blood cell in humans is the"),
    ("What normal body response",
     "The normal body response that raises your internal temperature to help "
     "kill viruses is"),
    ("What happens when the immune system mistakenly attacks",
     "When the immune system mistakenly attacks the body's own healthy tissues, "
     "the result is"),
    ("What specific organ system",
     "The specific organ system that influenza primarily attacks is the"),
    ("What long, flat parasites",
     "The long, flat parasites that can live in human intestines and are "
     "sometimes found in undercooked pork are"),
    ("What condition is caused by the overreaction",
     "The condition caused by the overreaction of the immune system to harmless "
     "substances like pollen is"),
    ("What waterborne disease",
     "The waterborne disease that causes severe diarrhea and dehydration is"),
    ("What bodily fluid contains enzymes",
     "The bodily fluid that contains enzymes that help break down bacteria in "
     "the mouth is"),
    ("What disease did John Snow map",
     "The disease John Snow mapped in 1854 to prove it was spread by "
     "contaminated water was"),
    ("What virus is responsible for causing AIDS",
     "The virus responsible for causing AIDS is"),
    ("What infectious disease is carried by the tsetse fly",
     "The infectious disease carried by the tsetse fly is"),
    ("What piece of protective gear",
     "The piece of protective gear worn over the nose and mouth by doctors "
     "during surgery is"),
    # POLAR - not sure we can use this one; see POLAR_ITEMS at the top
    ("Does sweating heavily cure a cold",
     "On the question of whether sweating heavily cures a cold, the answer is"),
    ("What historic disease",
     "The historic disease characterized by a 'bullseye' rash is"),
    ("How long can a cold virus typically survive",
     "A cold virus can typically survive on indoor surfaces for"),
    ("What viral disease is characterized by swollen salivary glands",
     "The viral disease characterized by swollen salivary glands is"),
    ("What over-the-counter medicine",
     "The over-the-counter medicine commonly used to reduce a fever is"),
    ("What is a medicated tablet",
     "A medicated tablet dissolved in the mouth to soothe a sore throat is called"),
    ("What respiratory disease was historically known",
     "The respiratory disease historically known as 'consumption' was"),
    # POLAR - not sure we can use this one; see POLAR_ITEMS at the top
    ("Is the 'stomach flu' the same virus",
     "On the question of whether the 'stomach flu' is the same virus as "
     "influenza, the answer is"),
    ("What highly contagious virus",
     "The highly contagious virus that causes a full-body rash and tiny white "
     "spots inside the mouth is"),
    ("What historical sailor's disease",
     "The historical sailor's disease caused by a lack of Vitamin C is"),
    ("What is the scientific term for an epidemic",
     "The scientific term for an epidemic that has spread across multiple "
     "countries or continents is"),
    ("What is the state of being resistant",
     "The state of being resistant to a specific disease is called"),
    ("What is the practice of isolating individuals",
     "The practice of isolating individuals who have been exposed to an "
     "infectious disease is called"),
    ("What is the most effective daily habit",
     "The most effective daily habit to prevent catching a cold is"),
    ("What itchy fungal infection",
     "The itchy fungal infection that commonly affects the skin between the "
     "toes is"),
    ("What season is the flu most common in",
     "The season the flu is most common in is"),
    ("What do the 'H' and 'N' stand for",
     "In the H1N1 virus, the 'H' and 'N' stand for"),
    # POLAR - not sure we can use this one; see POLAR_ITEMS at the top
    ("Does simply being in cold weather cause a cold",
     "On the question of whether simply being in cold weather causes a cold, "
     "the answer is"),
    # POLAR - not sure we can use this one; see POLAR_ITEMS at the top
    ("Can you cure the flu with antibiotics",
     "On the question of whether you can cure the flu with antibiotics, the "
     "answer is"),
    ("What mosquito-borne disease",
     "The mosquito-borne disease named after the yellowing of the skin it "
     "causes is"),
    ("What is the general name for a microscopic pathogen",
     "The general name for a microscopic pathogen that requires a living host "
     "cell to reproduce is"),
    ("What proteins does the immune system produce",
     "The proteins the immune system produces to neutralize foreign invaders are"),
    ("What deadly disease was officially eradicated",
     "The deadly disease officially eradicated globally in 1980 was"),
    ("What disease is caused by a toxin",
     "The disease caused by a toxin that affects the nervous system and is "
     "often associated with rusty nails is"),
    ("What is the term for a disease that can spread from animals",
     "The term for a disease that can spread from animals to humans is"),
    ("What animal famously carried the fleas",
     "The animal that famously carried the fleas that spread the bubonic plague "
     "was the"),
    ("What is a weakened or dead form",
     "A weakened or dead form of a pathogen introduced to the body is called"),
]
