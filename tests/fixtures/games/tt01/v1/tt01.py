# Test-only mini game (ACTION1 x3 = level done, ACTION2 = GAME_OVER). Not an ARC game.
from arcengine import ARCBaseGame, Level, Sprite
class Tt01(ARCBaseGame):
    def __init__(self, seed: int = 0):
        lv = [Level(sprites=[Sprite([[3]], name="p", x=i, y=0)], name=f"L{i}") for i in range(2)]
        super().__init__("tt01", lv, available_actions=[1, 2], seed=seed)
        self.n = 0
    def step(self):
        a = self.action.id.value
        if a == 1:
            self.n += 1
            if self.n == 3:
                self.n = 0
                self.next_level()
        elif a == 2:
            self.lose()
        self.complete_action()
