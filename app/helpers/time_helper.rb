module TimeHelper
  def self.exponential_drop_off(base: 2, power: 3)
    # absolute value
    base = base * -1 if base < 0
    power = power * -1 if power < 0

    base = 2 if base <= 1
    power = 2 if power <= 1

    base**power
  end
end