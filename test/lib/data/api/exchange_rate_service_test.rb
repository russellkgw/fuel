require 'test_helper'

class ExchangeRateServiceTest < ActiveSupport::TestCase
  test "mock service and save" do
    stub_request(:get, "https://openexchangerates.org/api/latest.json?app_id=test&base=USD").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'openexchangerates.org', 'User-Agent'=>'Ruby'}).
        to_return(status: 500).times(1).then.to_return(:status => 200, :body => '{"timestamp": 1481871613,"base": "USD","rates": {"YER": 250.109299,"ZAR": 13.94511,"ZMK": 5253.075255}}', :headers => {})

    TimeHelper.expects(:exponential_drop_off).returns(0.001)

    Data::Api::ExchangeRateService.call_service
    assert_equal(ExchangeRate.all.count, 1)
  end
end