# bundle exec rake api_services:make_call
namespace :api_services do
  desc "Call services"
  task make_call: :environment do
    Data::Api::ExchangeRateService.call_service
    Data::Api::OilPriceService.call_service
  end
end