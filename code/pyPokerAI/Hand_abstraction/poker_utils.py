from enum import Enum, IntEnum
import itertools
import numpy as np

class Suit(Enum):
    HEARTS = 'hearts'
    DIAMONDS = 'diamonds'
    CLUBS = 'clubs'
    SPADES = 'spades'


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class HandType(IntEnum):
    ROYAL_FLUSH = 9
    STRAIGHT_FLUSH = 8
    FOUR_OF_A_KIND = 7
    FULL_HOUSE = 6
    FLUSH = 5
    STRAIGHT = 4
    THREE_OF_A_KIND = 3
    TWO_PAIR = 2
    PAIR = 1
    HIGH_CARD = 0


class Card:
    """Class for representing a card."""

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __eq__(self, card2):
        """Two cards equal each other if their suit and rank are the same."""
        return self.suit == card2.suit and self.rank == card2.rank

    def __str__(self):
        return "%s of %s" % (self.rank.name, self.suit.name)

    def __hash__(self):
        if self.suit == Suit.HEARTS:
            result = 0
        elif self.suit == Suit.CLUBS:
            result = 20
        elif self.suit == Suit.SPADES:
            result = 40
        else:
            result = 60
        result += self.rank
        return result

    def __lt__(self, card2):
        return self.rank < card2.rank


class Hand:
    """Represents a standard 5-card poker hand."""

    def __init__(self, cards):
        if len(cards) != 5:
            raise ValueError('Hand does not contain 5 cards.')
        self.cards = set(cards)
        self.type = self._identify()

    def __str__(self):
        # TODO: Add example string output
        result = ''
        for card in self.cards:
            result += str(card) + '\n'
        return result

    def __gt__(self, hand2):
        """Returns True if the hand is a better hand than hand2."""
        # TODO: Implement
        if self.type > hand2.type:
            return True
        elif self.type < hand2.type:
            return False
        elif self.type == hand2.type:
            # NOTE: This ignores the case in which both hands have the same type
            # and rank. I think this is okay for now.
            return self.rank > hand2.rank

    def get_type(self):
        return self.type

    def get_rank(self):
        return self.rank

    def highest_rank(self):
        """Return the highest rank found in self.cards."""
        return max(self.cards).rank

    def has_rank(self, rank):
        """Returns True if the hand contains a card with the given rank.
        Input:
            rank - A member of the Rank enum class
        """
        for card in self.cards:
            if card.rank == rank:
                return True
        return False

    def _is_royal_flush(self):
        if self._is_straight_flush() and self.has_rank(Rank.ACE):
            self.rank = Rank.ACE
            return True
        else:
            return False

    def _is_straight_flush(self):
        if self._is_straight() and self._is_flush():
            self.rank = self.highest_rank()
            return True
        else:
            return False

    def _is_n_of_a_kind(self, n):
        """Returns true if the hand is a n-of-a-kind.
        Example: For that is a 4-of-a-kind, _is_n_of_a_kind(4) will be True.
        So will _is_n_of_a_kind(3) and _is_n_of_a_kind(2).
        Input:
            n - How many cards need to be the same rank to return True
        """
        for rank in Rank:
            counter = 0
            for card in self.cards:
                if card.rank == rank:
                    counter += 1
            if counter == n:
                self.rank = rank
                return True
        return False

    def _is_four_of_a_kind(self):
        return self._is_n_of_a_kind(4)

    def _is_full_house(self):
        if self._is_pair() and self._is_three_of_a_kind():
            self.rank = self.highest_rank()
            return True
        else:
            return False

    def _is_flush(self):
        for i, card in enumerate(self.cards):
            if i == 0:
                suit = card.suit
            else:
                if card.suit != suit:
                    return False
        self.rank = self.highest_rank()
        return True

    def _is_straight(self):
        sorted_cards = np.sort(list(self.cards))
        first_rank = sorted_cards[0].rank
        for i, card in enumerate(sorted_cards):
            if card.rank != first_rank + i:
                return False
        self.rank = self.highest_rank()
        return True

    def _is_three_of_a_kind(self):
        return self._is_n_of_a_kind(3)

    def _is_two_pair(self):
        if self._is_four_of_a_kind():
            return False
        num_pairs_found = 0
        for rank in Rank:
            counter = 0
            for card in self.cards:
                if card.rank == rank:
                    counter += 1
            if counter == 2:
                num_pairs_found += 1
                self.rank = rank  # Works because Rank iterates from low to high
        return num_pairs_found == 2

    def _is_pair(self):
        return self._is_n_of_a_kind(2)

    def _identify(self):
        if self._is_royal_flush():
            hand = HandType.ROYAL_FLUSH
        elif self._is_straight_flush():
            hand = HandType.STRAIGHT_FLUSH
        elif self._is_four_of_a_kind():
            hand = HandType.FOUR_OF_A_KIND
        elif self._is_full_house():
            hand = HandType.FULL_HOUSE
        elif self._is_flush():
            hand  = HandType.FLUSH
        elif self._is_straight():
            hand = HandType.STRAIGHT
        elif self._is_three_of_a_kind():
            hand = HandType.THREE_OF_A_KIND
        elif self._is_two_pair():
            hand = HandType.TWO_PAIR
        elif self._is_pair():
            hand = HandType.PAIR
        else:
            self.rank = self.highest_rank()
            hand = HandType.HIGH_CARD
        return hand


def get_deck():
    """Returns a standard unshuffled 52-card deck as a list of Card instances."""
    deck = []
    for suit in Suit:
        for rank in Rank:
            deck.append(Card(suit, rank))
    return deck


def read_card():
    """Returns a card instance inputted from the command line."""
    suit_is_valid = False
    while not suit_is_valid:
        suit_input = input('Suit: ').upper()
        for suit in Suit:
            if suit_input == suit.name:
                card_suit = suit
                suit_is_valid = True

    rank_is_valid = False
    while not rank_is_valid:
        rank_input = input('Rank: ').upper()
        for rank in Rank:
            if rank_input == rank.name:
                card_rank = rank
                rank_is_valid = True
    return Card(card_suit, card_rank)


def best_hand(cards):
    """Returns the best 5-card subset of the given cards.
    Input:
        cards - list of at least 5 Card instances
    Returns:
        Hand instance containing the best possible hand contained in cards.
    """
    return max(generate_all_hands(cards))


def generate_all_hands(cards):
    """Given a list of cards, returns every possible 5-card hand combination.
    Input:
        cards - list of at least 5 Card instances
    Returns:
        hands - list of Hand instances
    """
    if len(cards) < 5:
        raise ValueError('Too few cards')
    card_arrays = itertools.combinations(cards, 5)
    hands = []
    for card_array in card_arrays:
        new_hand = Hand(card_array)
        hands.append(new_hand)
    return hands