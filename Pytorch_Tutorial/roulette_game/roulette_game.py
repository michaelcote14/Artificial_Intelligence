import random


# todo get the total number of hits for each number in the last 50 spins

class RouletteGame:
    def __init__(self):
        self.consecutive_blacks = 0
        self.consecutive_reds = 0
        self.consecutive_odds = 0
        self.consecutive_evens = 0
        self.consecutive_lows = 0
        self.consecutive_highs = 0

        self.recent_numbers_list = [1]

        self.color_dictionary = {
            1: 'red', 2: 'black', 3: 'red', 4: 'black', 5: 'red',
            6: 'black', 7: 'red', 8: 'black', 9: 'red', 10: 'black',
            11: 'black', 12: 'red', 13: 'black', 14: 'red', 15: 'black',
            16: 'red', 17: 'black', 18: 'red', 19: 'red', 20: 'black',
            21: 'red', 22: 'black', 23: 'red', 24: 'black', 25: 'red',
            26: 'black', 27: 'red', 28: 'black', 29: 'black', 30: 'red',
            31: 'black', 32: 'red', 33: 'black', 34: 'red', 35: 'black',
            36: 'red'
        }

    def spin_wheel(self):
        number_spun = random.randint(1, 36)
        number_color = self.color_dictionary[number_spun]

        self.recent_numbers_list.append(number_spun)

        previous_number_spun = self.recent_numbers_list[0]
        previous_spin_color = self.color_dictionary[previous_number_spun]

        self.pattern_appender(number_spun, number_color, previous_number_spun, previous_spin_color)

        # Remove the oldest number from the list if the list is longer than 50
        if len(self.recent_numbers_list) > 50:
            self.recent_numbers_list.pop(0)

        return number_spun

    def pattern_appender(self, number_spun, number_spun_color, previous_number_spun, previous_spin_color):
        if number_spun <= 18:
            self.consecutive_lows += 1
            self.consecutive_highs = 0
        elif number_spun >= 19:
            self.consecutive_highs += 1
            self.consecutive_lows = 0

        if number_spun % 2 == 0:
            self.consecutive_evens += 1
            self.consecutive_odds = 0
        elif number_spun % 2 != 0:
            self.consecutive_odds += 1
            self.consecutive_evens = 0

        if number_spun_color == 'red':
            self.consecutive_reds += 1
            self.consecutive_blacks = 0
        elif number_spun_color == 'black':
            self.consecutive_blacks += 1
            self.consecutive_reds = 0

    def check_for_win(self, number_spun, neural_guess):
        # if neural guess = 0: low
        # if neural guess = 1: high
        # if neural guess = 2: even
        # if neural guess = 3: odd
        # if neural guess = 4: red
        # if neural guess = 5: black

        number_spun_color = self.color_dictionary[number_spun]

        if number_spun <= 18 and neural_guess == 0:
            return True
        elif number_spun > 18 and neural_guess == 1:
            return True
        elif number_spun % 2 == 0 and neural_guess == 2:
            return True
        elif number_spun % 2 != 0 and neural_guess == 3:
            return True
        elif number_spun_color == 'red' and neural_guess == 4:
            return True
        elif number_spun_color == 'black' and neural_guess == 5:
            return True
        else:
            return False

    def play_step(self, action):
        single_spin_winnings = 0

        number_spun = self.spin_wheel()

        if action[0] == 0:
            neural_guess = action[1]

            if self.check_for_win(number_spun, neural_guess) == True:
                single_spin_winnings += 1
                neural_reward = 1
            else:
                single_spin_winnings -= 1
                neural_reward = -1
        else:
            neural_reward = 0

        return neural_reward, single_spin_winnings


if __name__ == '__main__':
    roulette_game()
