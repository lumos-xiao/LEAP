"""
LEAP
====================================
"""
import copy
import numpy as np
from scipy.special import gamma as gamma
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared import utils
from textattack.shared.validators import transformation_consists_of_word_swaps


def sigmax(alpha):
    numerator = gamma(alpha + 1.0) * np.sin(np.pi * alpha / 2.0)
    denominator = gamma((alpha + 1) / 2.0) * alpha * np.power(2.0, (alpha - 1.0) / 2.0)

    return np.power(numerator / denominator, 1.0 / alpha)


def K(alpha):
    k = alpha * gamma((alpha + 1.0) / (2.0 * alpha)) / gamma(1.0 / alpha)
    k *= np.power(alpha * gamma((alpha + 1.0) / 2.0) / (gamma(alpha + 1.0) * np.sin(np.pi * alpha / 2.0)), 1.0 / alpha)

    return k


def C(alpha):
    x = np.array((0.75, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.95, 1.99))
    y = np.array(
        (2.2085, 2.483, 2.7675, 2.945, 2.941, 2.9005, 2.8315, 2.737, 2.6125, 2.4465, 2.206, 1.7915, 1.3925, 0.6089))

    return np.interp(alpha, x, y)


def vf(alpha):
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)

    x = x * sigmax(alpha)

    return x / np.power(np.abs(y), 1.0 / alpha)


def levy(alpha, gamma=1, n=1):
    w = 0;
    for i in range(0, n):
        v = vf(alpha)

        while v < -10:
            v = vf(alpha)

        w += v * ((K(alpha) - 1.0) * np.exp(-v / C(alpha)) + 1.0)

    z = 1.0 / np.power(n, 1.0 / alpha) * w * gamma

    return z


def get_one_levy(min, max):
    while 1:
        temp = levy(1.5, 1)
        if min <= temp <= max:
            break
        else:
            continue
    return temp


class LEAP(PopulationBasedSearch):

    def __init__(
            self, pop_size=60, max_iters=20, post_turn_check=True, max_turn_retries=20
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.post_turn_check = post_turn_check
        self.max_turn_retries = 20

        self._search_over = False
        self.omega_max = 0.8
        self.omega_min = 0.2
        self.c1_origin = 0.8
        self.c2_origin = 0.2
        self.v_max = 3.0


    def _greedy_perturb(self, pop_member, original_result):
        best_neighbors, prob_list = self._get_best_neighbors(
            pop_member.result, original_result
        )
        random_result = best_neighbors[np.argsort(prob_list)[-1]]
        pop_member.attacked_text = random_result.attacked_text
        pop_member.result = random_result
        return True

    def _equal(self, a, b):
        return -self.v_max if a == b else self.v_max

    def _turn(self, source_text, target_text, prob, original_text):
        """
        Based on given probabilities, "move" to `target_text` from `source_text`
        Args:
            source_text (PopulationMember): Text we start from.
            target_text (PopulationMember): Text we want to move to.
            prob (np.array[float]): Turn probability for each word.
            original_text (AttackedText): Original text for constraint check if `self.post_turn_check=True`.
        Returns:
            New `Position` that we moved to (or if we fail to move, same as `source_text`)
        """
        assert len(source_text.words) == len(
            target_text.words
        ), "Word length mismatch for turn operation."
        assert len(source_text.words) == len(
            prob
        ), "Length mismatch for words and probability list."
        len_x = len(source_text.words)

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_turn_retries + 1:
            indices_to_replace = []
            words_to_replace = []
            for i in range(len_x):
                if np.random.uniform() < prob[i]:
                    indices_to_replace.append(i)
                    words_to_replace.append(target_text.words[i])
            new_text = source_text.attacked_text.replace_words_at_indices(
                indices_to_replace, words_to_replace
            )
            indices_to_replace = set(indices_to_replace)
            new_text.attack_attrs["modified_indices"] = (
                                                                source_text.attacked_text.attack_attrs[
                                                                    "modified_indices"]
                                                                - indices_to_replace
                                                        ) | (
                                                                target_text.attacked_text.attack_attrs[
                                                                    "modified_indices"]
                                                                & indices_to_replace
                                                        )
            if "last_transformation" in source_text.attacked_text.attack_attrs:
                new_text.attack_attrs[
                    "last_transformation"
                ] = source_text.attacked_text.attack_attrs["last_transformation"]

            if not self.post_turn_check or (new_text.words == source_text.words):
                break

            if "last_transformation" in new_text.attack_attrs:
                passed_constraints = self._check_constraints(
                    new_text, source_text.attacked_text, original_text=original_text
                )
            else:
                passed_constraints = True

            if passed_constraints:
                break

            num_tries += 1

        if self.post_turn_check and not passed_constraints:
            # If we cannot find a turn that passes the constraints, we do not move.
            return source_text
        else:
            return PopulationMember(new_text)

    def _get_best_neighbors(self, current_result, original_result):
        """For given current text, find its neighboring texts that yields
        maximum improvement (in goal function score) for each word.

        Args:
            current_result (GoalFunctionResult): `GoalFunctionResult` of current text
            original_result (GoalFunctionResult): `GoalFunctionResult` of original text.
        Returns:
            best_neighbors (list[GoalFunctionResult]): Best neighboring text for each word
            prob_list (list[float]): discrete probablity distribution for sampling a neighbor from `best_neighbors`
        """
        current_text = current_result.attacked_text
        neighbors_list = [[] for _ in range(len(current_text.words))]
        transformed_texts = self.get_transformations(
            current_text, original_text=original_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            neighbors_list[diff_idx].append(transformed_text)

        best_neighbors = []
        score_list = []
        for i in range(len(neighbors_list)):
            if not neighbors_list[i]:
                best_neighbors.append(current_result)
                score_list.append(0)
                continue

            neighbor_results, self._search_over = self.get_goal_results(
                neighbors_list[i]
            )
            if not len(neighbor_results):
                best_neighbors.append(current_result)
                score_list.append(0)
            else:
                neighbor_scores = np.array([r.score for r in neighbor_results])
                score_diff = neighbor_scores - current_result.score
                best_idx = np.argmax(neighbor_scores)
                best_neighbors.append(neighbor_results[best_idx])
                score_list.append(score_diff[best_idx])

        prob_list = normalize(score_list)

        return best_neighbors, prob_list

    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        best_neighbors, prob_list = self._get_best_neighbors(
            initial_result, initial_result
        )
        population = []
        for _ in range(pop_size):
            random_result = np.random.choice(best_neighbors, 1, p=prob_list)[0]
            population.append(
                PopulationMember(random_result.attacked_text, random_result)
            )
        return population

    def perform_search(self, initial_result):
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
        # Initialize  velocities
        v_init = []
        v_init_rand = np.random.uniform(-self.v_max, self.v_max, self.pop_size)
        v_init_levy = []
        while 1:
            temp = levy(1.5, 1)
            if -self.v_max <= temp <= self.v_max:
                v_init_levy.append(temp)
            else:
                continue
            if len(v_init_levy) == self.pop_size:
                break
        for i in range(self.pop_size):
            if np.random.uniform(-self.v_max, self.v_max, ) < levy(1.5, 1):
                v_init.append(v_init_rand[i])
            else:
                v_init.append(v_init_levy[i])
        v_init = np.array(v_init)
        velocities = np.array(
            [
                [v_init[t] for _ in range(initial_result.attacked_text.num_words)]
                for t in range(self.pop_size)
            ]
        )

        global_elite = max(population, key=lambda x: x.score)
        if (
                self._search_over
                or global_elite.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        ):
            return global_elite.result

        local_elites = copy.copy(population)

        pop_fit_list = []
        for i in range(len(population)):
            pop_fit_list.append(population[i].score)
        pop_fit = np.array(pop_fit_list)
        fit_ave = round(pop_fit.mean(), 3)
        fit_min = pop_fit.min()

        # start iterations
        omega = []
        for i in range(self.max_iters):
            for k in range(len(population)):
                if population[k].score < fit_ave:
                    omega.append(self.omega_min + ((population[k].score - fit_min) *
                                                   (self.omega_max - self.omega_min)) /
                                 (fit_ave - fit_min))
                else:
                    omega.append(get_one_levy(0.5, 0.8))
            C1 = self.c1_origin - i / self.max_iters * (self.c1_origin - self.c2_origin)
            C2 = self.c2_origin + i / self.max_iters * (self.c1_origin - self.c2_origin)
            P1 = C1
            P2 = C2

            for k in range(len(population)):
                # calculate the probability of turning each word
                pop_mem_words = population[k].words
                local_elite_words = local_elites[k].words
                assert len(pop_mem_words) == len(
                    local_elite_words
                ), 

                for d in range(len(pop_mem_words)):
                    velocities[k][d] = omega[k] * velocities[k][d] + (1 - omega[k]) * (
                            self._equal(pop_mem_words[d], local_elite_words[d])
                            + self._equal(pop_mem_words[d], global_elite.words[d])
                    )
                
                turn_list = np.array([velocities[k]])
                turn_prob = softmax(turn_list)[0]

                if np.random.uniform() < P1:
                    population[k] = self._turn(
                        local_elites[k],
                        population[k],
                        turn_prob,
                        initial_result.attacked_text, )
                if np.random.uniform() < P2:
                    population[k] = self._turn(
                        global_elite,
                        population[k],
                        turn_prob,
                        initial_result.attacked_text,)

            # Check if there is any successful attack in the current population
            pop_results, self._search_over = self.get_goal_results(
                [p.attacked_text for p in population]
            )
            if self._search_over:
                # if `get_goal_results` gets cut short by query budget, resize population
                population = population[: len(pop_results)]
            for k in range(len(pop_results)):
                population[k].result = pop_results[k]

            top_member = max(population, key=lambda x: x.score)
            if (
                    self._search_over
                    or top_member.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_member.result

            # Mutation based on the current change rate
            for k in range(len(population)):
                change_ratio = population[k].attacked_text.words_diff_ratio(
                    local_elites[k].attacked_text
                )
                # Referred from the original source code
                p_change = 1 - 2 * change_ratio
                if np.random.uniform() < p_change:
                    self._greedy_perturb(population[k], initial_result)

                if self._search_over:
                    break

            # Check if there is any successful attack in the current population
            top_member = max(population, key=lambda x: x.score)
            if (
                    self._search_over
                    or top_member.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_member.result

            # Update the elite if the score is increased
            for k in range(len(population)):
                if population[k].score > local_elites[k].score:
                    local_elites[k] = copy.copy(population[k])

            if top_member.score > global_elite.score:
                global_elite = copy.copy(top_member)

        return global_elite.result

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word
        substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["pop_size", "max_iters", "post_turn_check", "max_turn_retries"]


def normalize(n):
    n = np.array(n)
    n[n < 0] = 0
    s = np.sum(n)
    if s == 0:
        return np.ones(len(n)) / len(n)
    else:
        return n / s



def softmax(x, axis=1):
    row_max = x.max(axis=axis)

    # Each element of the row needs to be subtracted from the corresponding maximum value, otherwise exp(x) will overflow, resulting in the inf case
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # Calculate the exponential power of e
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s