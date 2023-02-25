from datasets import load_from_disk
from datasets import load_dataset
import textattack
import transformers
from textattack import Attack
import textattack.search_methods.pso_leap as pso
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification, MaxModificationRate
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import WordSwapWordNet


ag = load_from_disk('ag_dataset')
model = transformers.AutoModelForSequenceClassification.from_pretrained('ag_model')
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

goal_function = UntargetedClassification(model_wrapper)
search_method = pso.ParticleSwarmOptimization(pop_size=60,max_iters=20,post_turn_check=True,max_turn_retries=20)
# (pop_size):  60
# (max_iters):  20
# (post_turn_check):  True
# (max_turn_retries):  20
constraints = [MaxModificationRate(max_rate=0.16),
               StopwordModification()]
transformation = WordSwapWordNet()  
attack = Attack(goal_function, constraints, transformation, search_method)
text = ag['test'][7599]['text']
label = ag['test'][7599]['label']
example_ori = [(text,label)]
for i in range(7599):
  text = ag['test'][i]['text']
  if text == '':
    continue
  label = ag['test'][i]['label']
  example_ori.append((text,label))
dataset = textattack.datasets.Dataset(example_ori)
attack_args = textattack.AttackArgs(num_examples=1000, log_to_txt="leap_ag.txt", log_to_csv='leap_ag.csv', disable_stdout=True, shuffle = True, random_seed=42)
attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
