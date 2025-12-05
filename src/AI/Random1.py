import random
import sys
import json
import os
import ast
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):
    # Learning constants
    ALPHA = 0.01  # Learning rate (how much to update values based on new information)
    GAMMA = 0.95  # Discount factor (how much to value future rewards vs immediate rewards)

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Random1")
        # Dictionary to store state utilities (O(1) lookup)
        self.state_utilities = {}
        # Try to load previous state utilities if they exist
        self.loadStateUtilities()
        # Keep track of the last state we updated and its reward
        self.last_state_key = None
        self.last_reward = 0.0
        # Fixed exploration rate for epsilon-greedy action selection
        self.epsilon = 0.1
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #hashState
    #Description: Extracts state features and creates a hashable tuple key
    #
    #Features extracted:
    #   my_food - Current player's food count
    #   opp_food - Opponent's food count
    #   food_lead_sign - 1 if ahead, -1 if behind, 0 if tied
    #   my_workers - Count of worker ants
    #   my_attackers - Count of soldier/r_soldier ants
    #   opp_attackers - Count of enemy soldier/r_soldier ants
    #   queen_danger - 1 if queen is in attack range of enemy attackers, 0 otherwise
    #   my_hill_health - Current player's anthill capture health
    #   enemy_hill_health - Opponent's anthill capture health
    #   my_queen_health - Current player's queen health, 0 if no queen
    #   enemy_queen_health - Opponent's queen health, 0 if no queen
    #   worker_to_deposit_dist - Distance from first carrying worker to nearest deposit (anthill/tunnel), -1 if no carrying worker/deposit
    #   worker_to_food_dist - Distance from first non-carrying worker to first food, -1 if no non-carrying worker/food
    #   attacker_to_enemy_queen_dist - Distance from first attacking unit to enemy queen, -1 if no attacker/enemy queen
    #   attacker_to_enemy_hill_dist - Distance from first attacking unit to enemy anthill, -1 if no attacker/enemy anthill
    #
    #Parameters:
    #   currentState - The state of the current game (GameState)
    #
    #Return: A hashable tuple representing the state features
    ##
    def hashState(self, currentState):
        myId = currentState.whoseTurn
        oppId = 1 - myId
        
        myInv = currentState.inventories[myId]
        oppInv = currentState.inventories[oppId]
        
        # Extract features
        my_food = myInv.foodCount
        opp_food = oppInv.foodCount
        
        # Food lead sign: 1 if ahead, -1 if behind, 0 if tied
        if my_food > opp_food:
            food_lead_sign = 1
        elif my_food < opp_food:
            food_lead_sign = -1
        else:
            food_lead_sign = 0
        
        # Count workers
        my_workers = len(getAntList(currentState, myId, (WORKER,)))
        
        # Count attackers (soldiers and r_soldiers)
        my_attackers = len(getAntList(currentState, myId, (SOLDIER, R_SOLDIER)))
        opp_attackers = len(getAntList(currentState, oppId, (SOLDIER, R_SOLDIER)))
        
        # Check queen danger: if any enemy attacker can reach my queen
        queen_danger = 0
        my_queen = myInv.getQueen()
        if my_queen is not None:
            queen_coords = my_queen.coords
            enemy_attackers = getAntList(currentState, oppId, (SOLDIER, R_SOLDIER))
            for attacker in enemy_attackers:
                attack_range = UNIT_STATS[attacker.type][RANGE]
                attackable_coords = listAttackable(attacker.coords, attack_range)
                if queen_coords in attackable_coords:
                    queen_danger = 1
                    break
        
        # Anthill health
        my_hill_health = 0
        my_hill = myInv.getAnthill()
        if my_hill is not None:
            my_hill_health = my_hill.captureHealth
        
        enemy_hill_health = 0
        enemy_hill = oppInv.getAnthill()
        if enemy_hill is not None:
            enemy_hill_health = enemy_hill.captureHealth
        
        # Queen health
        my_queen_health = 0
        if my_queen is not None:
            my_queen_health = my_queen.health
        
        enemy_queen_health = 0
        enemy_queen = oppInv.getQueen()
        if enemy_queen is not None:
            enemy_queen_health = enemy_queen.health
        
        # Distance from first carrying worker to nearest deposit
        worker_to_deposit_dist = -1
        my_workers_list = getAntList(currentState, myId, (WORKER,))
        
        # Find first carrying worker
        carrying_worker = None
        for worker in my_workers_list:
            if getattr(worker, 'carrying', False):
                carrying_worker = worker
                break
        
        if carrying_worker is not None:
            worker_coords = carrying_worker.coords
            
            # Find nearest deposit (anthill or tunnel)
            deposit_coords = []
            if my_hill is not None:
                deposit_coords.append(my_hill.coords)
            for tunnel in myInv.getTunnels():
                deposit_coords.append(tunnel.coords)
            
            if len(deposit_coords) > 0:
                min_dist = None
                for dep_coords in deposit_coords:
                    dist = approxDist(worker_coords, dep_coords)
                    if dist >= 0:
                        min_dist = dist if min_dist is None else min(min_dist, dist)
                if min_dist is not None:
                    worker_to_deposit_dist = min_dist
        
        # Distance from first non-carrying worker to first food
        worker_to_food_dist = -1
        
        # Find first non-carrying worker
        non_carrying_worker = None
        for worker in my_workers_list:
            if not getattr(worker, 'carrying', False):
                non_carrying_worker = worker
                break
        
        if non_carrying_worker is not None:
            worker_coords = non_carrying_worker.coords
            
            foods = getConstrList(currentState, NEUTRAL, (FOOD,))
            if len(foods) > 0:
                first_food = foods[0]
                dist = approxDist(worker_coords, first_food.coords)
                if dist >= 0:
                    worker_to_food_dist = dist
        
        # Distance from first attacking unit to enemy queen
        attacker_to_enemy_queen_dist = -1
        my_attackers_list = getAntList(currentState, myId, (SOLDIER, R_SOLDIER))
        if len(my_attackers_list) > 0:
            first_attacker = my_attackers_list[0]
            attacker_coords = first_attacker.coords
            
            if enemy_queen is not None:
                dist = approxDist(attacker_coords, enemy_queen.coords)
                if dist >= 0:
                    attacker_to_enemy_queen_dist = dist
        
        # Distance from first attacking unit to enemy anthill
        attacker_to_enemy_hill_dist = -1
        if len(my_attackers_list) > 0:
            first_attacker = my_attackers_list[0]
            attacker_coords = first_attacker.coords
            
            if enemy_hill is not None:
                dist = approxDist(attacker_coords, enemy_hill.coords)
                if dist >= 0:
                    attacker_to_enemy_hill_dist = dist
        
        # Return as hashable tuple
        return (my_food, opp_food, food_lead_sign, my_workers, 
                my_attackers, opp_attackers, queen_danger, 
                my_hill_health, enemy_hill_health, my_queen_health,
                enemy_queen_health, worker_to_deposit_dist,
                worker_to_food_dist, attacker_to_enemy_queen_dist,
                attacker_to_enemy_hill_dist)
    
    ##
    #hashStateAdversarial
    #Description: Extracts state features from the opponent's perspective and creates a hashable tuple key
    #
    #Features extracted (from opponent's perspective):
    #   my_food - Opponent's food count (from their perspective)
    #   opp_food - Current player's food count (from opponent's perspective)
    #   food_lead_sign - 1 if opponent ahead, -1 if behind, 0 if tied
    #   my_workers - Count of opponent's worker ants
    #   my_attackers - Count of opponent's soldier/r_soldier ants
    #   opp_attackers - Count of current player's soldier/r_soldier ants
    #   queen_danger - 1 if opponent's queen is in attack range of current player's attackers, 0 otherwise
    #   my_hill_health - Opponent's anthill capture health
    #   enemy_hill_health - Current player's anthill capture health
    #   my_queen_health - Opponent's queen health, 0 if no queen
    #   enemy_queen_health - Current player's queen health, 0 if no queen
    #   worker_to_deposit_dist - Distance from opponent's first carrying worker to nearest deposit, -1 if none
    #   worker_to_food_dist - Distance from opponent's first non-carrying worker to first food, -1 if none
    #   attacker_to_enemy_queen_dist - Distance from opponent's first attacking unit to current player's queen, -1 if none
    #   attacker_to_enemy_hill_dist - Distance from opponent's first attacking unit to current player's anthill, -1 if none
    #
    #Parameters:
    #   currentState - The state of the current game (GameState)
    #
    #Return: A hashable tuple representing the state features from opponent's perspective
    ##
    def hashStateAdversarial(self, currentState):
        # Swap perspective: opponent becomes "my", current player becomes "opp"
        myId = 1 - currentState.whoseTurn  # Opponent's ID
        oppId = currentState.whoseTurn      # Current player's ID
        
        myInv = currentState.inventories[myId]
        oppInv = currentState.inventories[oppId]
        
        # Extract features from opponent's perspective
        my_food = myInv.foodCount
        opp_food = oppInv.foodCount
        
        # Food lead sign: 1 if opponent ahead, -1 if behind, 0 if tied
        if my_food > opp_food:
            food_lead_sign = 1
        elif my_food < opp_food:
            food_lead_sign = -1
        else:
            food_lead_sign = 0
        
        # Count workers
        my_workers = len(getAntList(currentState, myId, (WORKER,)))
        
        # Count attackers (soldiers and r_soldiers)
        my_attackers = len(getAntList(currentState, myId, (SOLDIER, R_SOLDIER)))
        opp_attackers = len(getAntList(currentState, oppId, (SOLDIER, R_SOLDIER)))
        
        # Check queen danger: if any current player attacker can reach opponent's queen
        queen_danger = 0
        my_queen = myInv.getQueen()
        if my_queen is not None:
            queen_coords = my_queen.coords
            enemy_attackers = getAntList(currentState, oppId, (SOLDIER, R_SOLDIER))
            for attacker in enemy_attackers:
                attack_range = UNIT_STATS[attacker.type][RANGE]
                attackable_coords = listAttackable(attacker.coords, attack_range)
                if queen_coords in attackable_coords:
                    queen_danger = 1
                    break
        
        # Anthill health
        my_hill_health = 0
        my_hill = myInv.getAnthill()
        if my_hill is not None:
            my_hill_health = my_hill.captureHealth
        
        enemy_hill_health = 0
        enemy_hill = oppInv.getAnthill()
        if enemy_hill is not None:
            enemy_hill_health = enemy_hill.captureHealth
        
        # Queen health
        my_queen_health = 0
        if my_queen is not None:
            my_queen_health = my_queen.health
        
        enemy_queen_health = 0
        enemy_queen = oppInv.getQueen()
        if enemy_queen is not None:
            enemy_queen_health = enemy_queen.health
        
        # Distance from opponent's first carrying worker to nearest deposit
        worker_to_deposit_dist = -1
        my_workers_list = getAntList(currentState, myId, (WORKER,))
        
        # Find first carrying worker
        carrying_worker = None
        for worker in my_workers_list:
            if getattr(worker, 'carrying', False):
                carrying_worker = worker
                break
        
        if carrying_worker is not None:
            worker_coords = carrying_worker.coords
            
            # Find nearest deposit (anthill or tunnel)
            deposit_coords = []
            if my_hill is not None:
                deposit_coords.append(my_hill.coords)
            for tunnel in myInv.getTunnels():
                deposit_coords.append(tunnel.coords)
            
            if len(deposit_coords) > 0:
                min_dist = None
                for dep_coords in deposit_coords:
                    dist = approxDist(worker_coords, dep_coords)
                    if dist >= 0:
                        min_dist = dist if min_dist is None else min(min_dist, dist)
                if min_dist is not None:
                    worker_to_deposit_dist = min_dist
        
        # Distance from opponent's first non-carrying worker to first food
        worker_to_food_dist = -1
        
        # Find first non-carrying worker
        non_carrying_worker = None
        for worker in my_workers_list:
            if not getattr(worker, 'carrying', False):
                non_carrying_worker = worker
                break
        
        if non_carrying_worker is not None:
            worker_coords = non_carrying_worker.coords
            
            foods = getConstrList(currentState, NEUTRAL, (FOOD,))
            if len(foods) > 0:
                first_food = foods[0]
                dist = approxDist(worker_coords, first_food.coords)
                if dist >= 0:
                    worker_to_food_dist = dist
        
        # Distance from opponent's first attacking unit to current player's queen
        attacker_to_enemy_queen_dist = -1
        my_attackers_list = getAntList(currentState, myId, (SOLDIER, R_SOLDIER))
        if len(my_attackers_list) > 0:
            first_attacker = my_attackers_list[0]
            attacker_coords = first_attacker.coords
            
            if enemy_queen is not None:
                dist = approxDist(attacker_coords, enemy_queen.coords)
                if dist >= 0:
                    attacker_to_enemy_queen_dist = dist
        
        # Distance from opponent's first attacking unit to current player's anthill
        attacker_to_enemy_hill_dist = -1
        if len(my_attackers_list) > 0:
            first_attacker = my_attackers_list[0]
            attacker_coords = first_attacker.coords
            
            if enemy_hill is not None:
                dist = approxDist(attacker_coords, enemy_hill.coords)
                if dist >= 0:
                    attacker_to_enemy_hill_dist = dist
        
        # Return as hashable tuple (from opponent's perspective)
        return (my_food, opp_food, food_lead_sign, my_workers, 
                my_attackers, opp_attackers, queen_danger, 
                my_hill_health, enemy_hill_health, my_queen_health,
                enemy_queen_health, worker_to_deposit_dist,
                worker_to_food_dist, attacker_to_enemy_queen_dist,
                attacker_to_enemy_hill_dist)
    
    ##
    #getReward
    #Description: Returns reward for the current state
    #
    #Returns:
    #   +1.0 if current player has won
    #   -1.0 if current player has lost
    #   -0.01 for any other state (ongoing game)
    #
    #Parameters:
    #   currentState - The state of the current game (GameState)
    #
    def getReward(self, currentState):
        winner = getWinner(currentState)
        
        if winner == 1:
            # Current player has won
            return 1.0
        elif winner == 0:
            # Opponent has won
            return -1.0
        else:
            # No winner yet
            return -0.01
    
    ##
    #saveStateUtilities
    #Description: Saves the state_utilities dictionary to a file
    #
    #Parameters:
    #   filename - Optional filename. If not provided, uses default filename with "grahamm26"
    #
    def saveStateUtilities(self, filename=None):
        if filename is None:
            filename = "grahamm26_state_utilities.json"
        
        try:
            # Convert tuple keys to lists for JSON serialization
            json_data = {}
            for key, value in self.state_utilities.items():
                # Convert tuple key to list for JSON
                json_key = list(key)
                json_data[str(json_key)] = value
            
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"State utilities saved to {filename}")
        except Exception as e:
            print(f"Error saving state utilities to {filename}: {e}")
    
    ##
    #loadStateUtilities
    #Description: Loads the state_utilities dictionary from a file
    #
    #Parameters:
    #   filename - Optional filename. If not provided, uses default filename with "grahamm26"
    #
    def loadStateUtilities(self, filename=None):
        if filename is None:
            filename = "grahamm26_state_utilities.json"
        
        if not os.path.exists(filename):
            # File doesn't exist yet, start with empty dictionary
            return
        
        try:
            with open(filename, 'r') as f:
                json_data = json.load(f)
            
            # Convert list keys back to tuples
            self.state_utilities = {}
            for json_key, value in json_data.items():
                # Parse the string representation of the list back to a tuple
                # Using ast.literal_eval for safe evaluation
                key_list = ast.literal_eval(json_key)  # Convert string representation to list
                key_tuple = tuple(key_list)  # Convert list to tuple
                self.state_utilities[key_tuple] = value
            
            print(f"State utilities loaded from {filename}: {len(self.state_utilities)} states")
        except Exception as e:
            print(f"Error loading state utilities from {filename}: {e}")
            # If loading fails, start with empty dictionary
            self.state_utilities = {}
    
    ##
    #updateUtility
    #Description: Applies the TD(0) update to the stored utility of a state
    #
    #Parameters:
    #   prev_key   - Hash key of the previous state (tuple)
    #   reward     - Reward observed after leaving the previous state (float)
    #   next_key   - Hash key of the successor state (tuple) (optional)
    #   terminal   - True if the transition ended the game, so U(s') = 0
    ##
    def updateUtility(self, prev_key, reward, next_key=None, terminal=False):
        if prev_key is None:
            return
        
        if prev_key not in self.state_utilities:
            self.state_utilities[prev_key] = 0
        
        next_value = 0
        if not terminal and next_key is not None:
            if next_key not in self.state_utilities:
                self.state_utilities[next_key] = 0
            next_value = self.state_utilities[next_key]
        
        current_value = self.state_utilities[prev_key]
        td_target = reward + self.GAMMA * next_value
        self.state_utilities[prev_key] = current_value + self.ALPHA * (td_target - current_value)
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        # Hash the current state and register it (initialize to 0 if not seen)
        state_key = self.hashState(currentState)
        if state_key not in self.state_utilities:
            self.state_utilities[state_key] = 0
        
        # Update the utility of the previously visited state using the new evidence
        if self.last_state_key is not None:
            self.updateUtility(self.last_state_key, self.last_reward, state_key, terminal=False)
        
        # Remember this state and its reward for the next transition
        self.last_state_key = state_key
        self.last_reward = self.getReward(currentState)
        
        moves = listAllLegalMoves(currentState)
        
        # Filter out build moves if we already have 3+ ants
        numAnts = len(currentState.inventories[currentState.whoseTurn].ants)
        if numAnts >= 3:
            moves = [m for m in moves if m.moveType != BUILD]
        
        if len(moves) == 0:
            # Fallback: if no moves available, return END move
            return Move(END, None, None)
        
        # Evaluate each move by hashing the resulting state
        move_values = []
        for move in moves:
            # Get the state that would result from this move
            next_state = getNextState(currentState, move)
            # Hash the resulting state
            next_state_key = self.hashState(next_state)
            # Get the value (default to 0 if not seen)
            value = self.state_utilities.get(next_state_key, 0)
            move_values.append((value, move))
        
        # Epsilon-greedy action selection:
        #   With probability epsilon, pick a random move (explore)
        #   Otherwise, pick the move(s) with highest estimated utility (exploit)
        if random.random() < self.epsilon:
            selectedMove = moves[random.randint(0, len(moves) - 1)]
        else:
            # Find the maximum value
            max_value = max(move_values, key=lambda x: x[0])[0]
            
            # Get all moves with the maximum value (handle ties)
            best_moves = [move for value, move in move_values if value == max_value]
            
            # Randomly choose among the best moves (tie-breaking)
            selectedMove = best_moves[random.randint(0, len(best_moves) - 1)]
        
        return selectedMove
    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #printFeatureAnalytics
    #Description: Analyzes and prints statistics about feature values seen during the game
    #
    def printFeatureAnalytics(self):
        if len(self.state_utilities) == 0:
            print("No states recorded for analytics")
            return
        
        # Feature names matching the tuple order
        feature_names = [
            "my_food",
            "opp_food", 
            "food_lead_sign",
            "my_workers",
            "my_attackers",
            "opp_attackers",
            "queen_danger",
            "my_hill_health",
            "enemy_hill_health",
            "my_queen_health",
            "enemy_queen_health",
            "worker_to_deposit_dist",
            "worker_to_food_dist",
            "attacker_to_enemy_queen_dist",
            "attacker_to_enemy_hill_dist"
        ]
        
        # Collect all values for each feature
        feature_values = {name: [] for name in feature_names}
        
        for state_key in self.state_utilities.keys():
            for i, value in enumerate(state_key):
                feature_values[feature_names[i]].append(value)
        
        # Print statistics for each feature
        print("\n=== Feature Analytics ===")
        for name, values in feature_values.items():
            if len(values) == 0:
                continue
            
            # Calculate statistics
            unique_values = set(values)
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values) if len(values) > 0 else 0
            
            # Count value frequencies
            value_counts = {}
            for v in values:
                value_counts[v] = value_counts.get(v, 0) + 1
            
            # Print summary
            print(f"\n{name}:")
            print(f"  Range: [{min_val}, {max_val}]")
            print(f"  Average: {avg_val:.2f}")
            print(f"  Unique values: {len(unique_values)}")
            
            # Print most common values (top 5)
            sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            top_values = sorted_counts[:5]
            if len(top_values) > 0:
                print(f"  Most common values:")
                for val, count in top_values:
                    percentage = (count / len(values)) * 100
                    print(f"    {val}: {count} times ({percentage:.1f}%)")
    
    ##
    #printTrainingSummary
    #Description: Provides a concise overview of current learning progress
    #
    def printTrainingSummary(self):
        if len(self.state_utilities) == 0:
            print("Training summary unavailable: no stored states.")
            return
        
        values = list(self.state_utilities.values())
        avg_val = sum(values) / len(values)
        min_key = min(self.state_utilities, key=self.state_utilities.get)
        max_key = max(self.state_utilities, key=self.state_utilities.get)
        print("\n=== Training Summary ===")
        print(f"Visited states: {len(self.state_utilities)}")
        print(f"Average utility: {avg_val:.3f}")
        print(f"Min utility: {self.state_utilities[min_key]:.3f}  State: {min_key}")
        print(f"Max utility: {self.state_utilities[max_key]:.3f}  State: {max_key}")
    
    ##
    #registerWin
    #
    # 
    #
    def registerWin(self, hasWon):

        
        # Final TD update for the terminal transition
        terminal_reward = 1.0 if hasWon else -1.0
        self.updateUtility(self.last_state_key, terminal_reward, terminal=True)
        self.last_state_key = None
        self.last_reward = 0.0
        
        # Print the size of the state hash
        print(f"State hash size: {len(self.state_utilities)} unique states")
        self.printTrainingSummary()
        
        # Print feature analytics
        self.printFeatureAnalytics()
        
        # Save state utilities to file
        self.saveStateUtilities()
