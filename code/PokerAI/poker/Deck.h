#pragma once
#include <map>
#include <algorithm>
#include "Card.h"
#include "../util/Randint.h"
using namespace std;
const int total_cards = 52;
class Deck {
public:
	unsigned char* cards = new unsigned char[total_cards];
	unsigned char cur_index;
	Randint randi;
	Deck() {
		for (int i = 0; i < total_cards; i++)
			cards[i] = i;
		reset();
	}
	void reset(int _cur_index = 0) {
		cur_index = _cur_index;
		for (int i = 0; i < total_cards; i++)
			cards[i] = i;
	}
	void reset(unsigned char exclude[], int len) {
		cur_index = len;
		for (int i = len; i < total_cards; i++)
			cards[i] = i;
		for (int i = 0; i < len; i++) 
			cards[i] = exclude[i];
		bool vis[9] = { false };//找出小于9(len)的数,全部为true
		for (int i = 0; i < len; i++)
			if (exclude[i] < len) {
				if (!vis[exclude[i]])
					vis[exclude[i]] = true;
				else
					throw exception();
			}
		int k = 0;			///把没用过false的小于9(len)的数换到大于9的手牌位置
		for (int i = 0; i < len; i++)
			if (exclude[i] >= len) {
				while (vis[k]) k++;
				cards[exclude[i]] = k;
				vis[k++] = true;
			}
	}
	inline unsigned char deal_one_card() {
		int rs = (randi._rand() % (total_cards - cur_index)) + cur_index; //int rs = (rand() % (52 - cur_index)) + cur_index;
		swap(cards[cur_index], cards[rs]);
		return cards[cur_index++];
	}
};
