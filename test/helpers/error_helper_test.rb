require 'test_helper'

class ErrorHelperTest < ActiveSupport::TestCase
  test 'send error email' do
    ActionMailer::Base.deliveries.clear
    res = ErrorHelper.mail_error('test 1', 'test 2')

    assert_equal(1, ActionMailer::Base.deliveries.count)
    assert_equal('test2@example.com', res.from[0])
    assert_equal('test@example.com', res.to[0])
    assert_equal('Error in Fuel', res.subject)
  end
end