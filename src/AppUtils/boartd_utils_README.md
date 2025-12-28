
calc_board_connectivity:
Flop: 
1 - ascending order (789) - 3 straight possible
0.85 - one gap (78T) - 2 straights possible
0.7 - two gaps (79J) - 1 straight possible
0.7 - one big gap (78J) - 1 straight possible 
0.5 - 2 cards connected (278) - no straight possible
0.35 - 2 cards with gap (268) - no straight possible
0.2 - 2 cards with big gap (247) - no straight possible

Turn:
1 - ascending order (789T)
0.9 - one gap (789J) - one linear for a straight
0.8 - one big gap (789Q)
0.8 - two small gaps (68TJ)
0.7 - 3 small gaps (68TQ)
0.7 - one small gap, one big gap (689Q)
0.66 - one unrelated - 678Q
0.6 - 2 big gaps (69QK)

calc_board_flush:
this function calculates the flush potential on the board
Flop: 1 - monotone, 0.8 - flush draw
Turn: 0.3 - normal flush draw, 0.6 - double flush draw, 0.8 - 2 cards for flush, 1 - monotone (1 card for flush)
River: 1 - flush on table, 0.8 - one card for flush, 0.5 - 2 cards for flush