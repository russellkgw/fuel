require 'test_helper'

class TestHelperTest < ActiveSupport::TestCase
  test 'time helper exp drop off defualt args' do
    res = TimeHelper.exponential_drop_off
    assert_equal(8, res)
  end

  test 'time helper exp drop off, set base' do
    res = TimeHelper.exponential_drop_off(base: 3)
    assert_equal(27, res)
  end

  test 'time helper exp drop off, set power' do
    res = TimeHelper.exponential_drop_off(power: 4)
    assert_equal(16, res)
  end
end