class Data::Api::OilPriceService
  class Error < RuntimeError; end
  class IntegrationError < Error; end

  include TimeHelper
  include ErrorHelper

  RETRY_COUNT = 3

  def self.call_service
    new.call_service
  end

  def call_service(retry_count: RETRY_COUNT)
    call()
  rescue => e
    sleep(TimeHelper.exponential_drop_off(base: ((RETRY_COUNT - retry_count) + 2)))
    retry_count -= 1
    retry if retry_count > 0
    ErrorHelper.mail_error('Oil price service error: ', e.message)
  end

  def call
    prev_date = Date.today - 1   # We are looking for the closing rate of the previous day.
    return if (OilPrice.where(date: prev_date).any? || is_weekend_day?(prev_date))

    uri = URI("https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=#{Rails.application.secrets.oil_price_key}&start_date=#{prev_date.to_s}&end_date=#{prev_date.to_s}")
    service_request = Net::HTTP.get_response(uri)

    if service_request.response.is_a?(Net::HTTPSuccess)
      data = JSON.parse(service_request.body).to_h['dataset']['data']
      save_oil_price([0][1], prev_date) if data.any?
    else
      raise IntegrationError.new("Unable to successfully call oil price api, error: #{service_request.body.to_s}")
    end
  end

  def save_oil_price(value, date)
    OilPrice.create(currency: 'USD',
                    price: value,
                    date: date,
                    source: 'https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D')
  end

  def is_weekend_day?(day)
    day.saturday? || day.sunday?
  end
end