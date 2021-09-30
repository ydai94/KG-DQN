import numpy as np
import textworld
import re
import sys
import glob
import requests
import json
from textworld.core import EnvInfos

import subprocess
import time
def start_openie(install_path):
    print('Starting OpenIE from', install_path)
    subprocess.Popen(['java', '-mx8g', '-cp', '*', \
                      'edu.stanford.nlp.pipeline.StanfordCoreNLPServer', \
                      '-port', '9001', '-timeout', '15000', '-quiet'], cwd=install_path)
    time.sleep(1)

class NaiveAgent(textworld.Agent):
    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.actions = ["north", "south", "east", "west", "up", "down",
                        "look", "inventory", "take all", "YES", "wait",
                        "take", "drop", "eat", "attack"]

    def reset(self, env):
        env.display_command_during_render = True
        env.activate_state_tracking()

    def act(self, game_state, reward, done):

        action = self.rng.choice(self.actions)
        if action in ["take", "drop", "eat", "attack"]:
            words = game_state.feedback.split()  # Observed words.
            words = [w for w in words if len(w) > 3]  # Ignore most stop words.
            if len(words) > 0:
                action += " " + self.rng.choice(words)

        return action

class RandomAgent(textworld.Agent):
    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def reset(self, env):
        env.infos.admissible_commands = True
        env.display_command_during_render = True

    def act(self, game_state, reward, done):
        if game_state.admissible_commands is None:
            msg = "'--mode random-cmd' is only supported for generated games."
            raise NameError(msg)

        return self.rng.choice(game_state.admissible_commands)

class WalkthroughDone(NameError):
    pass


class WalkthroughAgent(textworld.Agent):
    """ Agent that simply follows a list of commands. """

    def __init__(self, commands=None):
        self.commands = commands

    def reset(self, env):
        env.display_command_during_render = True
        if self.commands is not None:
            self._commands = iter(self.commands)
            return  # Commands already specified.

        game_state = env.reset()
        if game_state.get("extra.walkthrough") is None:
            msg = "WalkthroughAgent is only supported for games that have a walkthrough."
            raise NameError(msg)

        # Load command from the generated game.
        self._commands = iter(game_state.get("extra.walkthrough"))

    def act(self, game_state, reward, done):
        try:
            action = next(self._commands)
        except StopIteration:
            raise WalkthroughDone()

        action = action.strip()  # Remove trailing \n, if any.
        return action


def test_agent(agent, game, out, max_step=1000, nb_episodes=5):
    info = EnvInfos(admissible_commands = True, description = True)
    env = textworld.start(game, info)  # Start the game.
    #print(game.split("/")[-1], end="")

    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores = [], []
    acts = set()
    for no_episode in range(nb_episodes):
        agent.reset(env)  # Tell the agent a new episode is starting.
        game_state = env.reset()  # Start new episode.

        reward = 0
        done = False
        for no_step in range(max_step):
            # print(game_state.description)

            command = agent.act(game_state, reward, done)

            out.write(game_state.description)
            out.write("Actions: " + str(game_state.admissible_commands) + '\n')
            acts.update(game_state.admissible_commands)
            out.write("Taken action:" + str(command))
            out.write('\n' + "---------" + '\n')
            game_state, reward, done = env.step(command)
            #env.render()

            # if no_step % 10 == 0:
            #    print(no_step, no_episode)

            if done:
                break

        # print("Done after {} steps. Score {}/1.".format(game_state.nb_moves, game_state.score))
        # print(".", end="")
        avg_moves.append(game_state.nb_moves)
        avg_scores.append(game_state.score)

    env.close()
    # print("  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / 1.".format(np.mean(avg_moves), np.mean(avg_scores)))
    # print(avg_moves)
    # exit()
    return acts


def call_stanford_openie(sentence):
    url = "http://localhost:9001/"
    querystring = {
        "properties": "%7B%22annotators%22%3A%20%22openie%22%7D",
        "pipelineLanguage": "en"}
    response = requests.request("POST", url, data=sentence, params=querystring)
    response = json.JSONDecoder().decode(response.text)
    return response

def generate_data(games, type):
        if type == 'collect':
            out = open("./random.txt", 'w')
            acts = set()
            for g in games:
                acts.update(test_agent(WalkthroughAgent(), game=g, out=out))
                acts.update(test_agent(RandomAgent(), game=g, out=out))
            out.close()

            out = open('./cleaned_random.txt', 'w')
            with open('./random.txt', 'r') as f:
                cur = []
                for line in f:
                    # print(line)
                    if line != '---------' and "Admissible actions:" not in str(line) and "Taken action:" not in str(
                            line):
                        cur.append(line)
                    else:
                        cur = [a.strip() for a in cur]
                        cur = ' '.join(cur).strip().replace('\n', '').replace('---------', '')
                        cur = re.sub("(?<=-\=).*?(?=\=-)", '', cur)
                        cur = cur.replace("-==-", '').strip()
                        cur = '. '.join([a.strip() for a in cur.split('.')])
                        out.write(cur + '\n')
                        cur = []
            out.close()

            input_file = open("./cleaned_random.txt", 'r')

            entities = set()
            relations = set()

            sents = input_file.read()
            sentences = sents.split('\n')
            triples = []
            for sent in sentences:
                res = call_stanford_openie(sent)['sentences']
                for r in res:
                    triples.append(r['openie'])
            try:
                
                for triple in triples:
                    for tr in triple:
                        h, r, t = tr['subject'], tr['relation'], tr['object']
                        entities.add(h)
                        entities.add(t)
                        relations.add(r)
                        print(' | ' + h + ', ' + r + ', ' + t,)
            except:
                print("OpenIE error")

            act_out = open('./act2id.txt', 'w')
            act_out.write(str({k: i for i, k in enumerate(acts)}))
            act_out.close()

            ent_out = open('./entity2id.tsv', 'w')
            rel_out = open('./relation2id.tsv', 'w')

            ent_out.write('\t' + str(0) + '\n')
            for i, e in enumerate(entities):
                ent_out.write('_'.join(e.split()) + '\t' + str(i+1) + '\n')

            ent_out.close()
            for i, r in enumerate(relations):
                rel_out.write('_'.join(r.split()) + '\t' + str(i) + '\n')
            rel_out.close()

        elif type == 'oracle':
            out = open("./oracle.txt", 'w')
            for g in games:
                test_agent(WalkthroughAgent(), game=g, out=out)
            out.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please supply directory with games and type.")
        exit()
    games = glob.glob(sys.argv[1] + '*.ulx')[:2]
    print(games)
    generate_data(games, sys.argv[2])
